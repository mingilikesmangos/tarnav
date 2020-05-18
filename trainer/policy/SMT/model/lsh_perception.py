import torch.nn as nn
import torch.nn.functional as F
from model.lshattention import LSHSelfAttention
from model.memory.memory import SceneMemory
class LSHAttblock(nn.Module):
    def __init__(self, cfg):
        super(LSHAttblock, self).__init__()
        '''
            def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 8,
                 add_local_attn_hash = False, causal = False, attn_chunks = 1,
                 random_rotations_per_head = False, attend_across_buckets = True,
                 allow_duplicate_attention = True, num_mem_kv = 0, one_value_head = False,
                 use_full_attn = False, full_attn_thres = None, return_attn = False,
                 post_attn_dropout = 0., dropout = 0., **kwargs):
                 
                 batch_size, seqlen, dim = qk.shape
        '''
        self.attn = LSHSelfAttention(dim=cfg.d_model, heads=cfg.n_head, bucket_size=cfg.lsh.bucket_size,
                                     n_hashes=cfg.lsh.n_hashes, add_local_attn_hash=cfg.lsh.add_local_attn_hash,
                                     causal=cfg.lsh.causal, attn_chunks=cfg.lsh.attn_chunks,
                                     random_rotations_per_head=cfg.lsh.random_rotations_per_head,
                                     attend_across_buckets=cfg.lsh.attend_across_buckets,
                                     allow_duplicate_attention=cfg.lsh.allow_duplicate_attention,
                                     num_mem_kv=cfg.lsh.num_mem_kv,
                                     one_value_head=cfg.lsh.one_value_head,
                                     use_full_attn=cfg.lsh.use_full_attn,
                                     full_attn_thres=cfg.lsh.full_attn_thres,
                                     return_attn=cfg.lsh.return_attn,
                                     post_attn_dropout=cfg.lsh.post_attn_dropout,
                                     dropout=cfg.lsh.dropout)
    #(self, x, keys = None, input_mask = None, input_attn_mask = None, context_mask = None):
    def forward(self, x, keys=None, input_mask = None, input_attn_mask = None, context_mask = None):
        x = self.attn(x, keys, input_mask, input_attn_mask, context_mask)
        return x


class LSHPerception(nn.Module):
    def __init__(self,cfg):
        super(LSHPerception, self).__init__()
        self.Encoder = LSHAttblock(cfg.attention)
        self.Decoder = LSHAttblock(cfg.attention)
        self.output_size = cfg.attention.d_model
        self.Memory = SceneMemory(cfg)
        self.Memory.reset_all()

    def cuda(self, device=None):
        super(LSHPerception, self).cuda(device)
        self.Memory.embed_network = self.Memory.embed_network.cuda()

    def act(self, obs, masks, mode='train'): # with memory
        obs['image'] = obs['image'] / 255.0 * 2 - 1.0
        embedded_memory, curr_embedding, pre_embedding, memory_masks = self.Memory.update_memory(obs, masks)
        C = self.Encoder(embedded_memory, input_mask=memory_masks)
        x = self.Decoder(curr_embedding, keys=C[:,1:], context_mask=memory_masks[:,1:])
        return x.squeeze(1), pre_embedding

    def forward(self, observations, memory_masks=None, mode='train'): # without memory
        if mode == 'pretrain':
            observations['image'] = observations['image'].float() / 255.0 * 2 - 1.0
            images, poses, prev_actions = observations['image'], observations['pose'], observations['prev_action']
            embedded_memory, curr_embedding, _ = self.Memory.embedd_observations(images, poses, prev_actions, memory_masks)
        else:
            batch_pre_embedding, batch_pose = observations
            embedded_memory, curr_embedding = self.Memory.embedd_with_pre_embeds(batch_pre_embedding,batch_pose, memory_masks)
        #if memory_masks is not None:
         #   memory_masks = memory_masks.unsqueeze(1)
        if memory_masks is not None:
            memory_masks = memory_masks.bool()
        C = self.Encoder(embedded_memory, input_mask=memory_masks)
        if memory_masks is not None:
            memory_masks = memory_masks[:,1:]
        x = self.Decoder(curr_embedding, keys=C[:,1:], context_mask=memory_masks)
        return x.squeeze(1)
