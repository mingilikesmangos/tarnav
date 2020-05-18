'''
-------> edited version for pytorch
'''

import torch
import torch.nn as nn

'''
Each image modality is embedded into 64-dimensional vectors using a modified ResNet-18 [18]. 
We reduce the numbers of filters of all convolutional layers by a factor of 4 and use stride of 1 for the first two convolutional layers. 
We remove the global pooling to better capture thespatial information and directly apply the fully-connectedlayer at the end.  
Both pose and action vectors are embedded using a single 16-dimensional fully-connected layer.
'''
'''
class Embedding(nn.Module):
    def __init__(self, cfg, embedding_network):
        super(Embedding, self).__init__()
        self.embed_image, self.embed_act, self.fc = embedding_network

    def forward(self, observations, prev_actions, masks):
        B = observations['panoramic_rgb'].shape[0]
        input_list = [observations['panoramic_rgb'].permute(0, 3, 1, 2) / 255.0,
                      observations['panoramic_depth'].permute(0, 3, 1, 2)]
        curr_obs = torch.cat(input_list, 1)

        goal_obs = observations['objectgoal'].permute(0, 3, 1, 2)
        batched_obs = torch.cat([curr_obs, goal_obs[:, :4]], 0)

        feats = self.embed_image(batched_obs)
        curr_feats, target_feats = feats.split(B)

        prev_actions_feats = self.embed_act(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )
        feats = self.fc(torch.cat([curr_feats.view(B,-1), target_feats.view(B,-1)],1))
        x = torch.cat([feats, prev_actions_feats],1)
        return x
'''


class SceneMemory(nn.Module):
    # B * M (max_memory_size) * E (embedding)
    def __init__(self, cfg, embedding_network) -> None:
        super(SceneMemory, self).__init__()
        self.B = cfg.NUM_PROCESSES#training.num_envs + cfg.training.valid_num_envs
        self.max_memory_size = cfg.memory.memory_size
        self.embedding_size = cfg.memory.embedding_size
        #self.embed_network = Embedding(cfg, embedding_network)
        self.gt_pose_buffer = torch.zeros([self.B, self.max_memory_size, 4],dtype=torch.float32).cuda()
        self.embedding_size_wo_pose = 512 + 32
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, self.embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size],dtype=torch.bool).cuda()
        self.reset_all()

    def freeze_embedding_network(self):
        self.embed_network.eval()


    def reset(self, reset_mask) -> None:
        assert(reset_mask.shape[0] == self.B)
        self.memory_buffer = self.memory_buffer * reset_mask.view(-1,1,1).float()
        self.memory_mask = self.memory_mask * reset_mask.view(-1,1).bool()
        self.gt_pose_buffer = self.gt_pose_buffer * reset_mask.view(-1,1,1).float()

    def reset_all(self) -> None:
        self.memory_buffer = torch.zeros([self.B, self.max_memory_size, self.embedding_size_wo_pose],dtype=torch.float32).cuda()
        self.memory_mask = torch.zeros([self.B, self.max_memory_size], dtype=torch.bool).cuda()
        self.gt_pose_buffer = torch.zeros([self.B, self.max_memory_size, 4], dtype=torch.float32).cuda()

    # as rollout gives memory, no need to handle memory anymore
    def update_memory(self, new_embedding, masks):
        if (masks == False).any(): self.reset(masks)
        self.memory_buffer = torch.cat([new_embedding['visual_features'], self.memory_buffer[:, :-1]], 1)
        # for debug
        #self.memory_buffer[:, 0, -1] = obs['pose'][:, -1]
        #self.memory_buffer[:, 0, -2] = obs['episode'].squeeze()
        self.memory_mask = torch.cat([torch.ones_like(masks, dtype=torch.bool), self.memory_mask[:,:-1]],1)

        length = self.memory_mask.sum(dim=1)
        max_length = int(length.max())

        return self.memory_buffer[:,:max_length],  self.memory_buffer[:,0:1], self.memory_mask[:,:max_length]

    def get_length(self):
        return self.memory_mask.sum(dim=1)

    def get_relative_poses(self, curr_pose, prev_poses):
        # curr_pose (x,y,orn,t)
        # shape B * 4
        curr_pose = curr_pose.unsqueeze(1) # shape B * 1 * 4
        curr_pose_x, curr_pose_y = curr_pose[:,:,0], curr_pose[:,:,1]
        gt_pose_x, gt_pose_y = prev_poses[:,:,0], prev_poses[:,:,1]
        curr_pose_yaw, gt_pose_yaw = curr_pose[:,:,2], prev_poses[:,:,2]

        del_x = gt_pose_x - curr_pose_x
        del_y = gt_pose_y - curr_pose_y
        th = torch.atan2(curr_pose_y, curr_pose_x)
        rel_x = del_x * torch.cos(th) - del_y * torch.sin(th)
        rel_y = del_x * torch.sin(th) + del_y * torch.cos(th)
        rel_yaw = gt_pose_yaw - curr_pose_yaw
        exp_t = torch.exp(-(prev_poses[:,0:1,3] - prev_poses[:,:,3]))

        relative_poses = torch.stack([rel_x, rel_y, torch.cos(rel_yaw), torch.sin(rel_yaw), exp_t],2)
        return relative_poses

    def embedd_observations(self, images, poses, prev_actions, memory_masks=None):
        # B * L * 3 * H * W : L will be 1
        #images, poses, prev_actions = images.squeeze(1), poses.squeeze(1), prev_actions.squeeze(1)

        relative_pose = self.get_relative_poses(poses[:, 0], poses)

        L = images.shape[1]
        embedded_memory = []
        for l in range(L):
            embeddings = [self.embed_network.embed_image(images[:,l]),
                          self.embed_network.embed_act(prev_actions[:,l]),
                          self.embed_network.embed_pose(relative_pose[:,l])]
            embeddings = torch.cat(embeddings, -1)
            embedded_memory.append(self.embed_network.final_embed(embeddings))
        if memory_masks is None:
            embedded_memory = torch.stack(embedded_memory, 1)
        else:
            embedded_memory = torch.stack(embedded_memory, 1) * memory_masks.view(-1,L,1)
        return embedded_memory, embedded_memory[:,0:1], memory_masks

    def embedd_with_pre_embeds(self, pre_embeddings, memory_masks=None):
        L = pre_embeddings.shape[1]
        #relative_pose = self.get_relative_poses(poses[:, 0], poses)
        #embedded_memory = torch.cat((pre_embeddings, relative_pose),1)
        embedded_memory = pre_embeddings
        if memory_masks is None:
            embedded_memory = embedded_memory
        else:
            embedded_memory = embedded_memory * memory_masks.view(-1,L,1)
        return embedded_memory, embedded_memory[:,0:1]
