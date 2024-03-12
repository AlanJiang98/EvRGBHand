from copy import deepcopy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision.models as models

# todo
# from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler

from src.modeling.fpn import build_fpn, build_position_encoding

from src.modeling.transformer import build_transformer, build_temporal_encoder


from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import scipy.stats as stats

class EvRGBmetro_vanilla(torch.nn.Module):
    def __init__(self, config):
        super(EvRGBmetro_vanilla, self).__init__()
        print('EvRGBmetro_vanilla')
        self.config = config
        self.num_joints = 21
        self.num_vertices_coarse = 195
        self.num_vertices_fine = 778
        self.aux_loss = config['model']['aux_loss']
        self.query_selection = config['model']['query_selection']
        hidden_dim = config['model']['hidden_dim']
        self.hidden_dim = hidden_dim
        self.training = not config['exper']['run_eval_only']
        backbone_num_channels = config['model']['backbone_num_channels']

        self.rgb_backbone = build_fpn(config, is_img=True)
        self.ev_backbone = build_fpn(config, is_img=False)

        #self.align_module = AlignBlock_nofusion(backbone_num_channels, backbone_num_channels)

        self.patch_size = config['model']['patch_size'] if 'patch_size' in config['model'] else 2
        patch_dim = hidden_dim * self.patch_size * self.patch_size

        self.to_patch_rgb = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        self.to_patch_ev = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        
        if self.config['model']['mask']:
            self.mask_ratio = self.config['model']['mask_ratio']
            mask_ratio_min = self.mask_ratio/2
            mask_ratio_max = self.mask_ratio*1.5
            mask_ratio_mu = self.mask_ratio
            mask_ratio_std = 0.25
            self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
            
            self.img_mask_token = nn.Parameter(torch.zeros(1,  hidden_dim))
            self.event_mask_token = nn.Parameter(torch.zeros(1,  hidden_dim))

        self.position_embedding = build_position_encoding(config['model']['pos_type'], config['model']['hidden_dim'])
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, 1, hidden_dim))
        
        transformer_config = dict(
                            d_model=config['model']['hidden_dim'],
                            nhead=config['model']['nhead'],
                            num_encoder_layers=config['model']['enc_layers'],
                            num_decoder_layers=config['model']['dec_layers'],
                            dim_feedforward=config['model']['dim_feedforward'],
                            dropout=config['model']['dropout'],
                            query_pos_update = config['model']['query_pos_update'] if 'query_pos_update' in config['model'] else False,
                            activation="relu",
                            return_intermediate_dec=True)
        
        self.transformer = build_transformer(transformer_config)

        # temporal attention 
        temporal_attention_config = dict(d_model=config['model']['hidden_dim'],
                                    nhead=config['model']['nhead'],
                                    num_encoder_layers=config['model']['temporal_attention_layers'],
                                    dim_feedforward=config['model']['dim_feedforward'],
                                    dropout=config['model']['dropout'],
                                    activation="relu")
        self.temporal_attention = build_temporal_encoder(temporal_attention_config)

        if self.query_selection:
            print('query selection. Carefully check the code!!!')
            self.get_heatmap = MLP(hidden_dim, hidden_dim, self.num_joints, 2)
            self.joints_to_vertices = torch.nn.Linear(self.num_joints, self.num_vertices_coarse)

            self.ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

            self.joint_token_embed = torch.nn.Embedding(self.num_joints, hidden_dim)
            self.vertex_token_embed = torch.nn.Embedding(self.num_vertices_coarse, hidden_dim)
        else:
            self.joint_token_embed = torch.nn.Embedding(self.num_joints, hidden_dim)
            self.vertex_token_embed = torch.nn.Embedding(self.num_vertices_coarse, hidden_dim)
            
            self.joint_pos_embed = torch.nn.Embedding(self.num_joints, hidden_dim)
            self.vertex_token_embed = torch.nn.Embedding(self.num_vertices_coarse, hidden_dim)

        self.xyz_regressor = MLP(hidden_dim, hidden_dim, 3, 3)

        self.learnable_upsample = config['model']['learnable_upsample']
        if self.learnable_upsample:
            self.mesh_sampler = None
            self.upsampling = torch.nn.Linear(self.num_vertices_coarse, self.num_vertices_fine)
        else:
            self.mesh_sampler = MeshSampler()
            self.upsampling = None
        
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.temporal_attention.parameters():
            param.requires_grad = True
    def forward(self, frames, return_att=True, decode_all=False):
        device = frames[0]['rgb'].device
        output = []

        time_step = len(frames)
        segments = len(frames[0]['ev_frames'])
        if segments > 1:
            rgb_all = torch.stack([frame['rgb'] for frame in frames], dim=0).permute(0, 1, 4, 2, 3)
            event_all = torch.stack([torch.stack(frame['ev_frames'], dim=0) for frame in frames], dim=0).permute(0, 1, 2, 5, 3, 4)
            rgb_all = repeat(rgb_all, 't b c h w -> t seg b c h w', seg=segments)
            rgb_all = rearrange(rgb_all, 't seg b c h w -> t (seg b) c h w')
            event_all = rearrange(event_all, 't seg b c h w -> t (seg b) c h w')
        else:
            rgb_all = torch.stack([frame['rgb'] for frame in frames], dim=0).permute(0, 1, 4, 2, 3)
            event_all = torch.stack([frame['ev_frames'][-1] for frame in frames], dim=0).permute(0, 1, 4, 2, 3)

        T, batch_size, inchs_rgb, height_rgb, width_rgb = rgb_all.shape
        _, _, inchs_ev, height_ev, width_ev = event_all.shape

        rgb_all = rearrange(rgb_all, 't b c h w -> (t b) c h w')
        event_all = rearrange(event_all, 't b c h w -> (t b) c h w')

        # extract features from rgb and event
        rgb_feat = self.rgb_backbone(rgb_all)[1]
        event_feat = self.ev_backbone(event_all)[1]

        #rgb_feat, event_feat = self.align_module(rgb_feat, event_feat)
        rgb_feat = self.to_patch_rgb(rgb_feat)
        rgb_src = rgb_feat.flatten(2).permute(2, 0, 1)
        event_feat = self.to_patch_ev(event_feat)
        event_src = event_feat.flatten(2).permute(2, 0, 1)
        
        # mask rgb and event
        if self.config['model']['mask'] and self.training:
            mask_rate = self.mask_ratio_generator.rvs(size=1)[0]
            L,tb,c = rgb_src.shape
            rgb_mask_len = int(L* mask_rate)
            rgb_noise = torch.randn((L, tb), device=device)
            sorted_rgb_noise, _ = torch.sort(rgb_noise, dim=0)
            rgb_cut_off = sorted_rgb_noise[rgb_mask_len-1:rgb_mask_len, :]
            rgb_mask = rgb_noise < rgb_cut_off
            rgb_src[rgb_mask] = self.img_mask_token

            mask_rate = self.mask_ratio_generator.rvs(size=1)[0]
            L,tb,c = event_src.shape
            event_mask_len = int(L* mask_rate)
            event_noise = torch.randn((L, tb), device=device)
            sorted_event_noise, _ = torch.sort(event_noise, dim=0)
            event_cut_off = sorted_event_noise[event_mask_len-1:event_mask_len, :]
            event_mask = event_noise < event_cut_off
            event_src[event_mask] = self.event_mask_token
            
        # position embedding
        bs,c,h_i,w_i = rgb_feat.shape
        bs,c,h_e,w_e = event_feat.shape
        
        assert bs == batch_size * time_step
        pos_rgb = self.position_embedding(bs, h_i, w_i, device)
        pos_event = self.position_embedding(bs, h_e, w_e, device)
        pos_rgb = pos_rgb.flatten(2).permute(2, 0, 1)
        pos_rgb = pos_rgb + self.modality_embedding[0]
        pos_event = pos_event.flatten(2).permute(2, 0, 1)
        pos_event = pos_event + self.modality_embedding[1]
        
        src = torch.cat([rgb_src, event_src], dim=0)
        pos = torch.cat([pos_rgb, pos_event], dim=0)


        # transformer 
        if self.query_selection:
            
            memory = self.transformer.encoder(src, pos = pos)

            # query selection
            h,w = h_e, w_e
            heatmap = self.get_heatmap(memory)
            heatmap = heatmap[:h*w,:,:] + heatmap[h*w:,:,:]
            heatmap = F.softmax(heatmap, dim=0)
            heatmap = rearrange(heatmap, '(h w) tb k -> k tb h w', h =h, w = w, k = self.num_joints)
            uu,vv = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float())
            uu = (uu + 0.5) / h
            vv = (vv + 0.5) / w
            uu = uu.to(device)
            vv = vv.to(device)
            joints_predictions = torch.stack((
                torch.sum(heatmap * vv[None,None,:,:], dim=(2,3)),
                torch.sum(heatmap * uu[None,None,:,:], dim=(2,3))), dim=2)
            vertices_predictions = self.joints_to_vertices(joints_predictions.transpose(0, 2)).transpose(0, 2)
            proposal = torch.cat([joints_predictions, vertices_predictions], dim=0) # (num_joints + num_vertices) X batch_size X  2

            query_sine_embed = gen_sineembed_for_position(proposal, N_steps=self.hidden_dim//2) # (num_joints + num_vertices) X batch_size X hidden_dim
            query_pos = self.ref_point_head(query_sine_embed) # (num_joints + num_vertices) X batch_size X hidden_dim

            jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0)
            tgt = jv_tokens.unsqueeze(1).repeat(1, bs, 1)
            #attention_mask = self.attention_mask.to(device)
            attention_mask = None

            hs = self.transformer.decoder(tgt, memory, query_pos=query_pos, pos=pos, tgt_mask=attention_mask)
            hs = hs.transpose(1, 2)
            joints_predictions = joints_predictions.transpose(0, 1)
            
        else:
            memory = self.transformer.encoder(src, pos = pos)

            jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0)
            tgt = jv_tokens.unsqueeze(1).repeat(1, bs, 1)
            
            query_pos = torch.cat([self.joint_pos_embed.weight, self.vertex_token_embed.weight], dim=0)
            query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)
            #attention_mask = self.attention_mask.to(device)
            attention_mask = None
            
            hs = self.transformer.decoder(tgt, memory, query_pos=query_pos, pos=pos, tgt_mask=attention_mask)
            hs = hs.transpose(1, 2)
            

        hs = rearrange(hs, 'L (T B) k d -> T L B k d', T=time_step, B=batch_size)
        #joints_predictions = rearrange(joints_predictions, '(T B) k d -> T B k d', T=time_step, B=batch_size)

        pred_3d_coordinates = self.xyz_regressor(hs) # T x L x batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, :, :, :self.num_joints,:] # T x L x batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:, :, :, self.num_joints:,:] # T x L x batch_size X num_vertices(coarse) X 3

        if self.learnable_upsample:
            pred_3d_vertices_fine = self.upsampling(pred_3d_vertices_coarse.transpose(3, 4)).transpose(3, 4) # T x L x batch_size X num_vertices(fine) X 3
            #pred_3d_vertices_fine_temporal = self.upsampling(pred_3d_vertices_coarse_temporal.transpose(2, 3)).transpose(2, 3) # T x batch_size X num_vertices(fine) X 3
        else:
            assert self.mesh_sampler is not None
            L,b,_,_ = pred_3d_vertices_coarse.shape
            pred_3d_vertices_coarse_tmp = rearrange(pred_3d_vertices_coarse, 'l b v c -> (l b) v c')
            pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_coarse_tmp)
            pred_3d_vertices_fine = rearrange(pred_3d_vertices_fine, '(l b) v c -> l b v c', l=L, b=b)

        for i in range(time_step):
            if segments > 1:
                output.append([])
                pred_3d_joints = rearrange(pred_3d_joints, 't l (seg b) v c -> l t seg b v c', seg=segments)
                pred_3d_vertices_coarse = rearrange(pred_3d_vertices_coarse, 't l (seg b) v c -> l t seg b v c', seg=segments)
                pred_3d_vertices_fine = rearrange(pred_3d_vertices_fine, 't l (seg b) v c -> t l seg b v c', seg=segments)
                for j in range(segments):
                    out = {'pred_3d_joints': pred_3d_joints[i,-1,j], 'pred_vertices_sub': pred_3d_vertices_coarse[i,-1,j], 'pred_vertices': pred_3d_vertices_fine[i,-1,j]}
                    if self.aux_loss:
                        out['aux_outputs'] = self._set_aux_loss(pred_3d_joints[i,:,j], pred_3d_vertices_coarse[i,:,j], pred_3d_vertices_fine[i,:,j])
                    output[-1].append(out)
                pass
            else:
                out = {'pred_3d_joints': pred_3d_joints[i,-1], 'pred_vertices_sub': pred_3d_vertices_coarse[i,-1], 'pred_vertices': pred_3d_vertices_fine[i,-1]}
                
                if self.aux_loss:
                    out['aux_outputs'] = self._set_aux_loss(pred_3d_joints[i], pred_3d_vertices_coarse[i], pred_3d_vertices_fine[i])
                
                output.append([out])
        return output
    @torch.jit.unused
    def _set_aux_loss(self, outputs_3d_joints, outputs_3d_vertices_sub, outputs_3d_vertices_fine):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_3d_joints': a, 'pred_vertices_sub': b,  'pred_vertices': c}
                                for a, b, c, in zip(outputs_3d_joints[:-1], outputs_3d_vertices_sub[:-1], outputs_3d_vertices_fine[:-1])]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def gen_sineembed_for_position(pos_tensor, N_steps=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(N_steps, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / N_steps)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)

    return pos





