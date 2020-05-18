import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def process_inputs_chunk(fn, chunks=1, dim=0):
    def inner_fn(*args, **kwargs):
        keys, values, len_args = kwargs.keys(), kwargs.values(), len(args)
        chunked_args = list(zip(*map(lambda x: x.chunk(chunks, dim=dim), list(args) + list(values))))
        all_args = map(lambda x: (x[:len_args], dict(zip(keys, x[len_args:]))), chunked_args)
        outputs = [fn(*c_args, **c_kwargs) for c_args, c_kwargs in all_args]
        return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))
    return inner_fn

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def default(val, default_val):
    return default_val if val is None else val

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

TOKEN_SELF_ATTN_VALUE = -5e4
class FullQKAttention(nn.Module):
    def __init__(self, causal = False, dropout = 0.):
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type(q.type())

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0)

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 4,
                 add_local_attn_hash = False, causal = False, attn_chunks = 1,
                 random_rotations_per_head = False, attend_across_buckets = True,
                 allow_duplicate_attention = True, num_mem_kv = 0, one_value_head = False,
                 use_full_attn = False, full_attn_thres = None, return_attn = False,
                 post_attn_dropout = 0., dropout = 0., **kwargs):
        super().__init__()
        assert dim % heads == 0, 'dimensions must be divisible by number of heads'

        self.dim = dim
        self.heads = heads
        self.attn_chunks = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim, bias = False)
        self.tov = nn.Linear(dim, v_dim, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.bucket_size = bucket_size
        self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, add_local_attn_hash = add_local_attn_hash, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, dropout = dropout, **kwargs)
        self.full_attn = FullQKAttention(causal = causal, dropout = dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        full_attn_thres = None if full_attn_thres == 'none' else full_attn_thres
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.callback = None

    def forward(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None):
        device, dtype = x.device, x.dtype
        b, t, e, h, m = *x.shape, self.heads, self.num_mem_kv

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(b, m, e)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        # merge head
        qk = qk.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)
        v = v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

        masks = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = mask[:, None, :].expand(b, h, kv_len).reshape(b * h, kv_len)
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = input_attn_mask[:, None, :, :].expand(b, h, t, t).reshape(b * h, t, t)
            masks['input_attn_mask'] = input_attn_mask

        attn_fn = self.lsh_attn if not use_full_attn else self.full_attn
        partial_attn_fn = partial(attn_fn, query_len = t)
        attn_fn_in_chunks = process_inputs_chunk(partial_attn_fn, chunks = self.attn_chunks)

        out, attn, buckets = attn_fn_in_chunks(qk, v, **masks)
        # split head
        out = out.view(b, h, t, -1).transpose(1, 2).contiguous().view(b, t, e)

        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))

        out = self.to_out(out)
        return self.post_attn_dropout(out)

class LSHAttention(nn.Module):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  add_local_attn_hash = False,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes
        self.add_local_attn_hash = add_local_attn_hash

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

    def hash_vectors(self, n_buckets, vecs): # vector들에게 hash function을 적용한다.

        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0 # hash fucntion  h(x) = argmax([ xR ; -xR]) 이기 때문에 반씩 구해야함.

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)
        # in paper, random matrix R shape is d_k, b/2 + (한꺼번에 n_hash 생성)
        # 여기서는 한꺼번에 n_hash 생성하기 위해 B, d_k, n, b/2 로 만듦

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
        #TODO: 왜 이게 random rotation이 되는지 모르겠어. random projection이라면 모를까, 이게 어떻게 unit circle projection이 되는거지???

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
        # batch * T * d_feature X batch * d_feature * n_hashes * hash_dim
        # ==> batch * n * seqlen * n_hashes * hash_dim

        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))

        return buckets # shape : B * seqen.. ? wh

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None):
        batch_size, seqlen, dim = qk.shape

        assert seqlen % (self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        query_len = default(query_len, seqlen)
        device = qk.device

        n_buckets = seqlen // self.bucket_size

        buckets = self.hash_vectors(n_buckets, qk) # buckets : hashed vectors, hashes

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen # 각 element마다 n개의 hash를 적용한다.

        if self.add_local_attn_hash: #TODO what is local attention
            local_buckets = torch.full((batch_size, seqlen), n_buckets, device=device, dtype=torch.long)
            buckets = torch.cat((buckets, local_buckets), dim=1)

        total_hashes = self.n_hashes + int(self.add_local_attn_hash)

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        #TODO buckets와 seqlen을 왜 또 곱하는가
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type(bq.type())

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :].clamp(max=query_len - 1)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad

        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets