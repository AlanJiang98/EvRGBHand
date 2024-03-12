from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.modeling._mano import Mesh as MeshSampler

from src.modeling.fpn import build_fpn, build_position_encoding
from src.modeling.fusion import SAFusionBlock
from src.modeling.transformer import build_transformer, build_temporal_encoder

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import scipy.stats as stats

class EvImHandNet(torch.nn.Module):
    def __init__(self, config):
        super(EvImHandNet, self).__init__()
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
        deformable_recurrent = config['model']['deformable_recurrent']

        self.rgb_backbone = build_fpn(config, is_img=True)
        self.ev_backbone = build_fpn(config, is_img=False)

        self.align_module = SAFusionBlock(backbone_num_channels, backbone_num_channels, recurrent=deformable_recurrent)

        self.patch_size = config['model']['patch_size'] if 'patch_size' in config['model'] else 2
        patch_dim = hidden_dim * self.patch_size * self.patch_size

        self.to_patch = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        if self.config['model']['mask']:
            self.mask_ratio = self.config['model']['mask_ratio']
            mask_ratio_min = self.mask_ratio/2
            mask_ratio_max = self.mask_ratio*1.5
            mask_ratio_mu = self.mask_ratio
            mask_ratio_std = 0.25
            self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
            
            self.img_mask_token = nn.Parameter(torch.zeros(1,  backbone_num_channels))
            self.event_mask_token = nn.Parameter(torch.zeros(1,  backbone_num_channels))

        self.position_embedding = build_position_encoding(config['model']['pos_type'], config['model']['hidden_dim'])
        
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
            self.get_heatmap = MLP(hidden_dim, hidden_dim, self.num_joints, 2)
            self.joints_to_vertices = torch.nn.Linear(self.num_joints, self.num_vertices_coarse)

            self.ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

            self.joint_token_embed = torch.nn.Embedding(self.num_joints, hidden_dim)
            self.vertex_token_embed = torch.nn.Embedding(self.num_vertices_coarse, hidden_dim)
        else:
            self.joint_token_embed = torch.nn.Embedding(self.num_joints, hidden_dim)
            self.vertex_token_embed = torch.nn.Embedding(self.num_vertices_coarse, hidden_dim)

        self.xyz_regressor = MLP(hidden_dim, hidden_dim, 3, 3)

        self.learnable_upsample = config['model']['learnable_upsample']
        if self.learnable_upsample:
            self.mesh_sampler = None
            self.upsampling = torch.nn.Linear(self.num_vertices_coarse, self.num_vertices_fine)
        else:
            self.mesh_sampler = MeshSampler()
            self.upsampling = None
        
        # attention mask
        # zeros_1 = torch.tensor(np.zeros((self.num_vertices_coarse, self.num_joints)).astype(bool))
        # zeros_2 = torch.tensor(np.zeros((self.num_joints, (self.num_joints + self.num_vertices_coarse))).astype(bool))
        # adjacency_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
        # adjacency_matrix_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
        # adjacency_matrix_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')
        # adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value,
        #                                             size=adjacency_matrix_size).to_dense()
        # temp_mask_1 = (adjacency_matrix == 0)
        # temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        # self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
        #self.attention_mask = None
        
    def forward(self, frames, **kwargs):
        device = frames[0]['rgb'].device
        output = []

        time_step = len(frames)
        
        segments = len(frames[0]['ev_frames'])
        time_step = time_step * segments
        
        if segments > 1:
            rgb_all = torch.stack([frame['rgb'] for frame in frames], dim=0).permute(0, 1, 4, 2, 3)
            event_all = torch.stack([torch.stack(frame['ev_frames'], dim=0) for frame in frames], dim=0).permute(0, 1, 2, 5, 3, 4)
            rgb_all = repeat(rgb_all, 't b c h w -> t seg b c h w', seg=segments)
            rgb_all = rearrange(rgb_all, 't seg b c h w -> (t seg) b c h w')
            event_all = rearrange(event_all, 't seg b c h w -> (t seg) b c h w')
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

        # mask rgb and event
        if self.config['model']['mask'] and self.training:
            mask_rate = self.mask_ratio_generator.rvs(size=1)[0]
            tb, c, rh, rw = rgb_feat.shape
            rgb_mask_len = int(rh* rw* mask_rate)
            rgb_flatten = rearrange(rgb_feat, 'tb c h w -> tb (h w) c')
            rgb_noise = torch.randn((tb, rh* rw), device=device)
            sorted_rgb_noise, _ = torch.sort(rgb_noise, dim=-1)
            rgb_cut_off = sorted_rgb_noise[:, rgb_mask_len-1:rgb_mask_len]
            rgb_mask = rgb_noise < rgb_cut_off
            rgb_flatten[rgb_mask] = self.img_mask_token
            rgb_feat = rearrange(rgb_flatten, 'tb (h w) c -> tb c h w', h=rh, w=rw)

            mask_rate = self.mask_ratio_generator.rvs(size=1)[0]
            tb, c, eh, ew = event_feat.shape
            event_mask_len = int(eh* ew* mask_rate)
            event_flatten = rearrange(event_feat, 'tb c h w -> tb (h w) c')
            event_noise = torch.randn((tb, eh* ew), device=device)
            sorted_event_noise, _ = torch.sort(event_noise, dim=-1)
            event_cut_off = sorted_event_noise[:, event_mask_len-1:event_mask_len]
            event_mask = event_noise < event_cut_off
            event_flatten[event_mask] = self.event_mask_token
            event_feat = rearrange(event_flatten, 'tb (h w) c -> tb c h w', h=eh, w=ew)

        fused_feat = self.align_module(rgb_feat, event_feat, time_step=time_step)
        fused_feat = self.to_patch(fused_feat)
        bs,c,h,w = fused_feat.shape
        assert bs == batch_size * time_step

        src = fused_feat.flatten(2).permute(2, 0, 1)

        pos = self.position_embedding(bs, h, w, device)
        pos = pos.flatten(2).permute(2, 0, 1)

        # transformer 
        if self.query_selection:
            memory = self.transformer.encoder(src, pos = pos)

            # temporal attention on transformer encoder feat
            memory_temporal = rearrange(memory, 'L (T B) D -> T B L D', T = time_step, B=batch_size)
            memory_temporal = self.temporal_attention(memory_temporal)
            memory = rearrange(memory_temporal, 'T B L D -> L (T B) D', T = time_step, B=batch_size)
            
            # query selection
            heatmap = self.get_heatmap(memory)
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
            joints_predictions = None
            jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0)
            attention_mask = None
            hs, memory = self.transformer(src, jv_tokens, pos, attention_mask=attention_mask)

            hs_out = hs[-1]
            hs_out = rearrange(hs_out, '(T B) J d -> T B J d', T=time_step, B=batch_size)
            hs = self.temporal_attention(hs_out)

        hs = rearrange(hs, 'L (T B) k d -> T L B k d', T=time_step, B=batch_size)
        joints_predictions = rearrange(joints_predictions, '(T B) k d -> T B k d', T=time_step, B=batch_size)

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
       
        if segments > 1:

            pred_3d_joints = rearrange(pred_3d_joints, '(t seg) l b v c -> t seg l b v c', seg=segments)
            pred_3d_vertices_coarse = rearrange(pred_3d_vertices_coarse, '(t seg) l b v c -> t seg l b v c', seg=segments)
            pred_3d_vertices_fine = rearrange(pred_3d_vertices_fine, '(t seg) l b v c -> t seg l b v c', seg=segments)
            joints_predictions = rearrange(joints_predictions, '(t seg) b v c -> t seg b v c', seg=segments)
            
            time_step = len(frames)
            for i in range(time_step):
                output.append([])
                for j in range(segments):
                    out = {'pred_3d_joints': pred_3d_joints[i,j,-1], 'pred_vertices_sub': pred_3d_vertices_coarse[i,j,-1], 'pred_vertices': pred_3d_vertices_fine[i,j,-1], 'joints_hm': joints_predictions[i,j]}
                    
                    if self.aux_loss:
                        out['aux_outputs'] = self._set_aux_loss(pred_3d_joints[i,j], pred_3d_vertices_coarse[i,j], pred_3d_vertices_fine[i,j])
                
                    output[-1].append(out)
        else:
            for i in range(time_step):
                out = {'pred_3d_joints': pred_3d_joints[i,-1], 'pred_vertices_sub': pred_3d_vertices_coarse[i,-1], 'pred_vertices': pred_3d_vertices_fine[i,-1], 'joints_hm': joints_predictions[i]}
                
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





