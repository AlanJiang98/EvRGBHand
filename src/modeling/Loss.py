import numpy as np
import torch
import torch.nn.functional as F
from src.modeling._mano import MANO
from src.modeling._mano import Mesh as MeshSampler
from pytorch3d.renderer import (BlendParams, HardFlatShader, MeshRasterizer,
                                MeshRenderer, PointLights,
                                RasterizationSettings, PerspectiveCameras,
                                TexturesVertex,
                                SoftSilhouetteShader)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt


class Loss(torch.nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.mesh_sampler = MeshSampler()
        self.init_criterion()
        self.init_silouette_render()

    def init_silouette_render(self):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        event_raster_settings = RasterizationSettings(
            image_size=(self.config['exper']['bbox']['event']['size'], self.config['exper']['bbox']['event']['size']),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=20,
            perspective_correct=False
        )
        rgb_raster_settings = RasterizationSettings(
            image_size=(self.config['exper']['bbox']['rgb']['size'], self.config['exper']['bbox']['rgb']['size']),
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=20,
            perspective_correct=False
        )
        self.event_silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=event_raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
        self.rgb_silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=rgb_raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )

    def init_criterion(self):
        # define loss function (criterion) and optimizer
        self.criterion_2d_joints = torch.nn.MSELoss(reduction='none').cuda(self.config['exper']['device'])
        self.criterion_3d_joints = torch.nn.SmoothL1Loss(reduction='none', beta=0.01).cuda(self.config['exper']['device'])
        self.criterion_vertices = torch.nn.SmoothL1Loss(reduction='none', beta=0.01).cuda(self.config['exper']['device'])
        
    def get_3d_joints_loss(self, gt_3d_joints, pred_3d_joints, mask):
        gt_root = gt_3d_joints[:, 0, :]
        gt_3d_aligned = gt_3d_joints - gt_root[:, None, :]
        pred_root = pred_3d_joints[:, 0, :]
        pred_3d_aligned = pred_3d_joints - pred_root[:, None, :]
        
        if mask.any():
            return (self.criterion_3d_joints(gt_3d_aligned, pred_3d_aligned)[mask]).mean()
        else:
            return torch.tensor(0, device=gt_3d_joints.device)

    # todo check the confidence of each joints
    def get_2d_joints_loss(self, gt_3d_joints, pred_3d_joints, mask, meta_data):
        if mask.any():
            K_event_ori = meta_data['K_event']
            K_event = K_event_ori.clone()
            R_event = meta_data['R_event']
            t_event = meta_data['t_event']
            K_event[:, :2, 2] = K_event_ori[:, :2, 2] - meta_data['lt_evs'][-1]
            K_event[:, :2] = K_event_ori[:, :2] *  meta_data['sc_evs'][-1][..., None, None]
            pred_kp = (K_event @ (R_event @ pred_3d_joints.transpose(2, 1) + t_event.reshape(-1, 3, 1))).transpose(2, 1)
            pred_joints_2d = pred_kp[:, :, :2] / pred_kp[:, :, 2:]
            pred_joints_2d_norm = pred_joints_2d / self.config['exper']['bbox']['event']['size']

            gt_kp = (K_event @ (R_event @ gt_3d_joints.transpose(2, 1) + t_event.reshape(-1, 3, 1))).transpose(2, 1)
            gt_joints_2d = gt_kp[:, :, :2] / gt_kp[:, :, 2:]
            gt_joints_2d_norm = gt_joints_2d / self.config['exper']['bbox']['event']['size']

            loss = (self.criterion_2d_joints(gt_joints_2d_norm, pred_joints_2d_norm)[mask]).mean()
        else:
            loss = torch.tensor(0, device=gt_3d_joints.device)
        return loss

    def get_2d_joints_loss_with_cam(self, gt_2d_joints, pred_3d_joints, camera, mask, meta_data):
        X = pred_3d_joints
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_2d = camera[:,:,:1] * X_trans
        # shape = X_trans.shape
        # X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)

        loss = (self.criterion_2d_joints(gt_2d_joints, X_2d)[mask]).mean()
        return loss

    def get_vertices_loss(self, gt_vertices, pred_vertices, mask):
        
        pred_vertices_aligned = pred_vertices
        loss = self.criterion_vertices(gt_vertices, pred_vertices_aligned)
        if mask.any():
            return (loss[mask]).mean()
        else:
            return torch.tensor(0, device=gt_vertices.device)

    def get_edge_loss(self, gt_vertices, pred_vertices, mask):
        face = self.mano_layer.faces
        face = torch.tensor(face.astype(np.int64)).to(pred_vertices.device)
        coord_out = pred_vertices[mask]
        coord_gt = gt_vertices[mask]
        if len(coord_gt) > 0:
            d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

            d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
            d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

            diff1 = torch.abs(d1_out - d1_gt)
            diff2 = torch.abs(d2_out - d2_gt) 
            diff3 = torch.abs(d3_out - d3_gt) 
            edge_diff = torch.cat((diff1, diff2, diff3),1)
            loss = edge_diff.mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device) 

        return loss

    def get_normal_loss(self, gt_vertices, pred_vertices, mask):
        face = self.mano_layer.faces
        face = torch.tensor(face.astype(np.int64)).to(pred_vertices.device)
        coord_out = pred_vertices[mask]
        coord_gt = gt_vertices[mask]
        if len(coord_gt) > 0:
            v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
            v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
            v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:] # batch_size X num_faces X 3
            v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

            v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
            v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
            normal_gt = torch.cross(v1_gt, v2_gt, dim=2) # batch_size X num_faces X 3
            normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

            cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            loss = torch.cat((cos1, cos2, cos3),1).mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(pred_vertices.device) 

        return loss
    # todo  DICE loss for segmentation here
    def get_dice_loss(self, seg1, seg2, mask):
        seg1_ = seg1.flatten(start_dim=1)
        seg2_ = seg2.flatten(start_dim=1)
        intersection = (seg1_ * seg2_).sum(-1).sum()
        score = (2. * intersection + 1e-6) / (seg1_.sum() + seg2_.sum() + 1e-6)
        return 1 - score


    def get_loss_mano(self, meta_data, preds):
        bbox_valid = meta_data['bbox_valid']
        mano_valid = meta_data['mano_valid'] * bbox_valid
        joints_3d_valid = meta_data['joints_3d_valid'] * bbox_valid
        manos = meta_data['mano']

        gt_dest_mano_output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        gt_dest_vertices_sub = self.mesh_sampler.downsample(gt_dest_mano_output.vertices)
        gt_dest_root = gt_dest_mano_output.joints[:, 0, :]
        gt_dest_3d_joints_aligned = gt_dest_mano_output.joints - gt_dest_root[:, None, :]
        gt_dest_vertices_aligned = gt_dest_mano_output.vertices - gt_dest_root[:, None, :]
        gt_dest_vertices_sub_aligned = gt_dest_vertices_sub - gt_dest_root[:, None, :]

        pred_3d_joints_from_mesh = self.mano_layer.get_3d_joints(preds['pred_vertices'])

        pred_3d_joints_root = preds['pred_3d_joints'][:, :1, :]
        pred_3d_joints_from_mesh_root = pred_3d_joints_from_mesh[:, :1, :]

        pred_vertices_coarse_aligned = preds['pred_vertices_sub'] - pred_3d_joints_from_mesh_root
        pred_vertices_fine_aligned = preds['pred_vertices'] - pred_3d_joints_from_mesh_root
        pred_3d_joints_aligned = preds['pred_3d_joints'] - pred_3d_joints_root
        pred_3d_joints_from_mesh_aligned = pred_3d_joints_from_mesh - pred_3d_joints_from_mesh_root

        pred_3d_joints_cor = pred_3d_joints_aligned + gt_dest_root[:, None, :]
        pred_3d_joints_from_mesh_cor = pred_3d_joints_from_mesh_aligned + gt_dest_root[:, None, :]

        loss_3d_joints = self.get_3d_joints_loss(gt_dest_3d_joints_aligned,
                                                    pred_3d_joints_aligned,
                                                    mano_valid)
        loss_3d_joints_reg = self.get_3d_joints_loss(gt_dest_3d_joints_aligned,
                                                        pred_3d_joints_from_mesh_aligned,
                                                        mano_valid)
        loss_vertices = self.get_vertices_loss(gt_dest_vertices_aligned,
                                                pred_vertices_fine_aligned,
                                                mano_valid)
        loss_vertices_sub = self.get_vertices_loss(gt_dest_vertices_sub_aligned,
                                                    pred_vertices_coarse_aligned,
                                                    mano_valid)
        
        if 'joints_hm' in preds.keys() and preds['joints_hm'] is not None:
            loss_joints_hm = F.mse_loss(preds['joints_hm'][mano_valid], meta_data['2d_joints_mesh'][mano_valid])
        else:
            loss_joints_hm = torch.tensor(0., device=loss_3d_joints.device)
        
        ratio = 1
        loss_sum = self.config['exper']['loss']['vertices'] *  loss_vertices + \
                    self.config['exper']['loss']['vertices_sub'] *  loss_vertices_sub + \
                    self.config['exper']['loss']['3d_joints'] * loss_3d_joints + \
                    self.config['exper']['loss']['3d_joints_from_mesh'] * loss_3d_joints_reg + \
                    self.config['exper']['loss']['joints_hm'] * loss_joints_hm

        loss_dict = {
            'loss_3d_joints': loss_3d_joints,
            'loss_3d_joints_reg': loss_3d_joints_reg,
            'loss_vertices': loss_vertices,
            'loss_vertices_sub': loss_vertices_sub,
            'loss_joints_hm': loss_joints_hm,
        }
        if 'pred_cam' in preds.keys() and preds['pred_cam'] is not None:
            camera = preds['pred_cam']
            loss_2d_joints_cam = self.get_2d_joints_loss_with_cam(meta_data['2d_joints_mesh'],
                                                                pred_3d_joints_aligned,
                                                                camera,
                                                                mano_valid,
                                                                meta_data)  
            loss_2d_joints_cam_reg = self.get_2d_joints_loss_with_cam(meta_data['2d_joints_mesh'],
                                                                pred_3d_joints_from_mesh_aligned,
                                                                camera,
                                                                mano_valid,
                                                                meta_data)
            loss_sum += self.config['exper']['loss']['2d_joints'] * loss_2d_joints_cam + \
                        self.config['exper']['loss']['2d_joints_from_mesh'] * loss_2d_joints_cam_reg
            
            loss_dict.update({
                'loss_2d_joints': loss_2d_joints_cam,
                'loss_2d_joints_reg': loss_2d_joints_cam_reg,
            })
        
        return loss_sum, loss_dict
    
    def forward(self, meta_data, preds):
        steps = len(meta_data)
        device = preds[0][-1]['pred_vertices'].device
        loss_sum = torch.tensor([0.], device=device, dtype=torch.float32)
        loss_items = {}
        for step in range(steps):
            bbox_valid = meta_data[step]['bbox_valid']
            mano_valid = meta_data[step]['mano_valid'] * bbox_valid
            joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid
            super_3d_valid = meta_data[step]['supervision_type'] == 0
            super_2d_valid = torch.logical_not(super_3d_valid)
            manos = meta_data[step]['mano']

            if self.config['model']['arch'] == 'eventhands':
                loss_hand_pose = F.mse_loss(preds[step][-1]['pred_hand_pose'], manos['hand_pose'].squeeze(dim=1))
                loss_shape = F.mse_loss(preds[step][-1]['pred_shape'], manos['shape'].squeeze(dim=1))
                loss_rot_pose = F.mse_loss(preds[step][-1]['pred_rot_pose'], manos['rot_pose'].squeeze(dim=1))

                loss_items.update({
                    'loss_hand_pose_' + str(step): loss_hand_pose,
                    'loss_shape_' + str(step): loss_shape,
                    'loss_rot_pose_' + str(step): loss_rot_pose,
                })

                loss_sum += loss_hand_pose * 5. + loss_shape * 5. + loss_rot_pose * 20
                continue

            loss_sum_step, loss_items_step = self.get_loss_mano(meta_data[step], preds[step][-1])
            loss_sum += loss_sum_step

            loss_items.update({
                'loss_sum_'+str(step): loss_sum_step,
                'loss_vertices_'+str(step): loss_items_step['loss_vertices'],
                'loss_vertices_sub_'+str(step): loss_items_step['loss_vertices_sub'],
                'loss_3d_joints_'+str(step): loss_items_step['loss_3d_joints'],
                'loss_3d_joints_reg_'+str(step): loss_items_step['loss_3d_joints_reg'],
                'loss_joints_hm_'+str(step): loss_items_step['loss_joints_hm'],
            })

            if 'aux_outputs' in preds[step][-1]:
                for i, aux_output in enumerate(preds[step][-1]['aux_outputs']):
                    loss_sum_step, loss_items_step = self.get_loss_mano(meta_data[step], aux_output)
                    loss_sum += loss_sum_step * self.config['exper']['loss']['aux_loss_weight']
                    loss_items.update({
                        f'loss_sum_aux_{step}_{i}': loss_sum_step,
                        f'loss_vertices_aux_{step}_{i}': loss_items_step['loss_vertices'],
                        f'loss_vertices_sub_aux_{step}_{i}': loss_items_step['loss_vertices_sub'],
                        f'loss_3d_joints_aux_{step}_{i}': loss_items_step['loss_3d_joints'],
                        f'loss_3d_joints_reg_aux_{step}_{i}': loss_items_step['loss_3d_joints_reg'],
                    })
        return loss_sum, loss_items
