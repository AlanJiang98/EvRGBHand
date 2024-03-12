# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
FastMETRO model.
"""
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer

from src.modeling.misc import  is_main_process
import torchvision
import math
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 dilation: bool = False,
                 is_img=True):
        super(Backbone, self).__init__()
        norm_layer = FrozenBatchNorm2d if not is_img else nn.BatchNorm2d
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        #assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)

        if name in ('resnet18', 'resnet34'):
            print("Using resnet18/34")
            self.backbone_channels = [64, 128, 256, 512]
        else:
            self.backbone_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return [c2, c3, c4, c5]


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, bs, h, w, device):
        ones = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        y_embed = ones.cumsum(1, dtype=torch.float32)
        x_embed = ones.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(pos_type, hidden_dim):
    N_steps = hidden_dim // 2
    if pos_type == 'sine':
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError("not supported {pos_type}")

    return position_embedding

class FastMETRO_Hand_Network(nn.Module):
    """FastMETRO for 3D hand mesh reconstruction from a single RGB image"""
    def __init__(self, config):

        super().__init__()
        self.config = config
        self.num_joints = 21
        self.num_vertices_coarse = 195
        self.num_vertices_fine = 778

        self.modality = config['model']['modality'] if 'modality' in config['model'] else 'rgb'
        
        self.backbone = Backbone(config['model']['backbone'], is_img=(self.modality=='rgb'))
        
        transformer_config_1 = dict(
                            d_model=config['model']['hidden_dim_1'],
                            nhead=config['model']['nhead'],
                            num_encoder_layers=config['model']['enc_layers'],
                            num_decoder_layers=config['model']['dec_layers'],
                            dim_feedforward=config['model']['dim_feedforward_1'],
                            dropout=config['model']['dropout'],
                            activation="relu",
                            return_intermediate_dec=True)
        transformer_config_2 = dict(
                            d_model=config['model']['hidden_dim_2'],
                            nhead=config['model']['nhead'],
                            num_encoder_layers=config['model']['enc_layers'],
                            num_decoder_layers=config['model']['dec_layers'],
                            dim_feedforward=config['model']['dim_feedforward_2'],
                            dropout=config['model']['dropout'],
                            activation="relu",
                            return_intermediate_dec=True)

        # build transformers
        self.transformer_1 = build_transformer(transformer_config_1)
        self.transformer_2 = build_transformer(transformer_config_2)

        # dimensionality reduction
        self.dim_reduce_enc_img = nn.Linear(config['model']['hidden_dim_1'], config['model']['hidden_dim_2'])
        self.dim_reduce_dec = nn.Linear(config['model']['hidden_dim_1'], config['model']['hidden_dim_2'])
        
        # token embeddings
        self.joint_token_embed = nn.Embedding(self.num_joints, config['model']['hidden_dim_1'])
        self.vertex_token_embed = nn.Embedding(self.num_vertices_coarse, config['model']['hidden_dim_1'])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=config['model']['pos_type'], hidden_dim=config['model']['hidden_dim_1'])
        self.position_encoding_2 = build_position_encoding(pos_type=config['model']['pos_type'], hidden_dim=config['model']['hidden_dim_2'])
        # estimators
        self.xyz_regressor = nn.Linear(config['model']['hidden_dim_2'], 3)
        self.upsampling = torch.nn.Linear(self.num_vertices_coarse, self.num_vertices_fine)
        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(self.backbone.backbone_channels[-1], config['model']['hidden_dim_1'], kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(np.zeros(( self.num_vertices_coarse,  self.num_joints)).astype(bool)) 
        zeros_2 = torch.tensor(np.zeros(( self.num_joints, ( self.num_joints +  self.num_vertices_coarse))).astype(bool)) 
        adjacency_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
    
    def forward(self, frames, return_att=True, decode_all=False):
        
        if self.modality == 'rgb':
            images = frames[0]['rgb'].permute(0, 3, 1, 2)
        elif self.modality == 'event':
            segments = len(frames[0]['ev_frames'])
            images_all = [ ev_frames.permute(0, 3, 1, 2) for ev_frames in frames[0]['ev_frames'] ]
            images = torch.cat(images_all, dim=0)
            #images = frames[0]['ev_frames'][-1].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"not supported modality {self.modality}")
        device = images.device
        batch_size = images.size(0)

        # preparation
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1) # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(device) # (num_joints + num_vertices) X (num_joints + num_vertices)
        
        # extract image features through a CNN backbone
        img_features = self.backbone(images)[-1] # batch_size X 2048 X 7 X 7
        _, _, h, w = img_features.shape
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        
        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 128 

        # first transformer encoder-decoder
        jv_features_1, enc_img_features_1 = self.transformer_1(img_features, jv_tokens, pos_enc_1, attention_mask=attention_mask)
        jv_features_1 = jv_features_1[-1].transpose(0,1)
        enc_img_features_1 = enc_img_features_1.transpose(0,1)
        
        # progressive dimensionality reduction
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1) # 49 X batch_size X 128 
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1) # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        jv_features_2, _ = self.transformer_2(reduced_enc_img_features_1, reduced_jv_features_1, pos_enc_2, attention_mask=attention_mask)
        jv_features_2 = jv_features_2[-1] 

        # estimators
        pred_3d_coordinates = self.xyz_regressor(jv_features_2) # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:,:self.num_joints,:] # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:,self.num_joints:,:] # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_fine = self.upsampling(pred_3d_vertices_coarse.transpose(1,2)).transpose(1, 2) # batch_size X num_vertices(fine) X 3
        
        if self.modality == 'event' and segments > 1:
            # split batch
            pred_3d_joints = pred_3d_joints.chunk(segments, dim=0)
            pred_3d_vertices_coarse = pred_3d_vertices_coarse.chunk(segments, dim=0)
            pred_3d_vertices_fine = pred_3d_vertices_fine.chunk(segments, dim=0)

            output = []
            output.append([])
            for seg in range(segments):
                out = {'pred_3d_joints': pred_3d_joints[seg], 'pred_vertices_sub': pred_3d_vertices_coarse[seg], 'pred_vertices': pred_3d_vertices_fine[seg]}
                output[0].append(out)
        else:
            out = {'pred_3d_joints': pred_3d_joints, 'pred_vertices_sub': pred_3d_vertices_coarse, 'pred_vertices': pred_3d_vertices_fine}
            output = []
            output.append([out])

        return output