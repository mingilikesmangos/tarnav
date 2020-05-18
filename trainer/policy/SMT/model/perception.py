import torch.nn as nn
import torch.nn.functional as F
from trainer.policy.SMT.model.attention import MultiHeadAttention
from trainer.policy.SMT.model.memory.memory import SceneMemory
class Attblock(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, dropout=0.1):
        super(Attblock, self).__init__()
        self.att_residual = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y, mask=None):
        enc_output, enc_slf_attn = self.att_residual(X,Y,Y, mask)
        H = self.layer_norm(enc_output)
        residual = H
        x = self.fc2(F.relu(self.fc1(H)))
        x = self.dropout(x)
        x += residual
        return x


class Perception(nn.Module):
    def __init__(self,cfg, embedding_network):
        super(Perception, self).__init__()
        self.Encoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.Decoder = Attblock(cfg.attention.n_head, cfg.attention.d_model, cfg.attention.d_k, cfg.attention.d_v, cfg.attention.dropout)
        self.output_size = cfg.attention.d_model
        self.Memory = SceneMemory(cfg, embedding_network)
        self.Memory.reset_all()

    def cuda(self, device=None):
        super(Perception, self).cuda(device)
        self.Memory.embed_network = self.Memory.embed_network.cuda()

    # this function is only used when eval
    def act(self, embeddings, masks, mode='train'): # with memory
        embedded_memory, curr_embedding, memory_masks = self.Memory.update_memory(embeddings,masks)
        C = self.Encoder(embedded_memory, embedded_memory, memory_masks.unsqueeze(1))
        x = self.Decoder(curr_embedding, C, memory_masks.unsqueeze(1))
        return x.squeeze(1), curr_embedding

    def forward(self, observations, memory_masks=None, mode='train'): # without memory
        #embedded_memory, curr_embedding = self.Memory.embedd_with_pre_embeds(observations['embeddings'], observations['memory_masks'])
        embedded_memory, memory_masks = observations['embeddings'], observations['memory_masks']
        curr_embedding = observations['embeddings'][:,0:1]
        memory_max_length = memory_masks.sum(dim=1).max().long()
        #print(memory_max_length)
        mem_mask = memory_masks[:,:memory_max_length].squeeze(2).unsqueeze(1)
        C = self.Encoder(embedded_memory[:,:memory_max_length], embedded_memory[:,:memory_max_length], mem_mask)
        x = self.Decoder(curr_embedding, C, mem_mask)
        return x.squeeze(1)
