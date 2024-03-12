"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
from os.path import dirname
import os.path as op

import code
import json
import time
import datetime
import warnings
import torch
import tqdm
import torchvision.models as models
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
import copy
import imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.modeling.Loss import Loss
from src.modeling._mano import MANO, Mesh
from src.datasets.build import make_hand_data_loader


from src.utils.joint_indices import indices_change
from src.utils.comm import synchronize, setup_for_distributed, is_main_process, get_rank, get_world_size, all_gather, to_device
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, SmoothedValue
from src.utils.renderer import Render
from src.configs.config_parser import ConfigParser
from src.utils.metric_pampjpe import get_alignMesh, compute_similarity_transform_batch
from fvcore.nn import FlopCountAnalysis
from src.modeling.build import build_model

def save_checkpoint(model, optimizer,scheduler, config, epoch,iteration, scaler=None):
    latest_checkpoint_path = op.join(config['exper']['output_dir'], 'latest.pth')
    save_checkpoint_path = op.join(config['exper']['output_dir'], f'{epoch:03d}_{iteration}.pth')
    if not is_main_process():
        return latest_checkpoint_path
    if not os.path.exists(config['exper']['output_dir']):
        os.makedirs(config['exper']['output_dir'])
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'model': model_to_save.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    if epoch % 10 == 0:
        torch.save(checkpoint, save_checkpoint_path)

    checkpoint.update({'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict()})
    if scaler is not None:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, latest_checkpoint_path)

    return latest_checkpoint_path


def run(config, train_dataloader, EvRGBStereo_model, Loss):
    save_config = copy.deepcopy(config)
    save_config['exper']['device'] = 'cuda'
    ConfigParser.save_config_dict(save_config, config['exper']['output_dir'], 'train.yaml')
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // config['exper']['num_train_epochs']
    
    model_without_ddp = EvRGBStereo_model
    if config['exper']['distributed']:
        EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
            EvRGBStereo_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=True,
        )
        Loss = torch.nn.parallel.DistributedDataParallel(
            Loss, device_ids=[int(os.environ["LOCAL_RANK"])],

        )
    
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, config['exper']['lr_backbone_names']) and p.requires_grad],
            "lr": config['exper']['lr'],
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, config['exper']['lr_backbone_names']) and p.requires_grad],
            "lr": config['exper']['lr_backbone'],
        }
    ]

    optimizer = torch.optim.AdamW(params=param_dicts,
                                 lr=config['exper']['lr'],
                                 betas=(0.9, 0.999),
                                 weight_decay=1e-4)

    # todo add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['exper']['num_train_epochs'], eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() if config['exper']['amp'] else None
    
    start_training_time = time.time()
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = SmoothedValue()
    last_epoch = 0
    iteration = 0
    # if config['exper']['resume_checkpoint'] != None and config['exper']['resume_checkpoint'] != 'None':
            
    #         latest_dict = torch.load(config['exper']['resume_checkpoint'], map_location='cpu')
    #         optimizer.load_state_dict(latest_dict["optimizer"])
    #         scheduler.load_state_dict(latest_dict["scheduler"])
    #         # last_epoch = latest_dict["epoch"]
    #         # iteration = latest_dict["iteration"]
    #         del latest_dict
    #         gc.collect()
    #         torch.cuda.empty_cache()
    
    EvRGBStereo_model.train()
    for i, (frames, meta_data) in enumerate(train_dataloader):
        
        iteration += 1
        epoch = iteration // iters_per_epoch
        data_time.update(time.time() - end)

        device = 'cuda'
        batch_size = frames[0]['rgb'].shape[0]
        frames = to_device(frames, device)
        meta_data = to_device(meta_data, device)
        with torch.cuda.amp.autocast(enabled=config['exper']['amp']):
            preds = EvRGBStereo_model(frames, return_att=True, decode_all=False)
            loss_sum, loss_items = Loss(meta_data, preds)

        # back prop
        optimizer.zero_grad()
        if config['exper']['amp']:
            scaler.scale(loss_sum).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(EvRGBStereo_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(EvRGBStereo_model.parameters(), 1.0)
            optimizer.step()
        
        if last_epoch != epoch:
            scheduler.step()
            
        log_losses.update(loss_sum.item(), batch_size)
        
        last_epoch = epoch
        if is_main_process():
            for key in loss_items.keys():
                tf_logger.add_scalar(key, loss_items[key], iteration)

        batch_time.update(time.time() - end)
        end = time.time()

        if (iteration-1) % config['utils']['logging_steps'] == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if is_main_process():
                print(
                    ' '.join(
                        ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                    ).format(eta=eta_string, ep=epoch, iter=iteration,
                             memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                    + '   compute: {:.4f}, data: {:.4f}, lr: {:.6f} loss: {:.6f} loss_joints: {:.6f} loss_vertices {:.6f}'.format(
                        batch_time.avg,
                        data_time.avg,
                        optimizer.param_groups[0]['lr'],
                        log_losses.avg,
                        loss_items.get('loss_3d_joints_0').item() if 'loss_3d_joints_0' in loss_items.keys() else 0,
                        loss_items.get("loss_vertices_0").item() if 'loss_vertices_0' in loss_items.keys() else 0, 
                    )
                )

        if iteration % iters_per_epoch == 0:
            save_checkpoint(EvRGBStereo_model, optimizer, scheduler, config, epoch, iteration, scaler=scaler)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    print('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    save_checkpoint(EvRGBStereo_model, optimizer, scheduler, config, epoch, iteration, scaler=scaler)


def update_errors_list(errors_list, mpjpe_eachjoint_eachitem, index):
    if len(mpjpe_eachjoint_eachitem) != 0:
        for i, id in enumerate(index):
            errors_list[id].append(mpjpe_eachjoint_eachitem[i].detach().cpu())


def print_metrics(errors, metric='MPJPE', f=None):
    errors_all = 0
    joints_all = 0
    # remove the fast motion results
    count = 0
    scene_errors = 0
    scene_items = 0

    for key, value in errors.items():
        count += 1
        if value is None:
            if f is None:
                print('{} is {}'.format(key, None))
            else:
                f.write('{} is {}'.format(key, None) + '\n')
        else:
            #valid_items = value != 0.
            valid_items = torch.ones_like(value)
            mpjpe_tmp = torch.sum(value) / (torch.sum(valid_items)+1e-6)
            errors_all += torch.sum(value)
            joints_all += torch.sum(valid_items)
            scene_errors += torch.sum(value)
            scene_items += torch.sum(valid_items)
            if f is None:
                print('{} {} is {}'.format(metric, key, mpjpe_tmp))
            else:
                f.write('{} {} is {}'.format(metric, key, mpjpe_tmp) + '\n')
            if count % 2 == 0:
                if f is None:
                    print('Scene Average is {}'.format(scene_errors / (scene_items + 1e-6)))
                else:
                    f.write('Scene Average is {}'.format(scene_errors / (scene_items + 1e-6)) + '\n')
                scene_errors = 0
                scene_items = 0
    mpjpe_all = errors_all / (joints_all+1e-6)
    if f is None:
        print('{} all is {}'.format(metric, mpjpe_all))
    else:
        f.write('{} all is {}'.format(metric, mpjpe_all) + '\n')


def print_sequence_error(mpjpe_eachjoint_eachitem, metric, seq_type, dir):
    x = np.arange(0, mpjpe_eachjoint_eachitem.shape[0])
    mpjpe_pre = torch.mean(mpjpe_eachjoint_eachitem, dim=1).detach().cpu().numpy() / 1000.
    plt.plot(x, mpjpe_pre, label=metric)
    plt.plot(x, mpjpe_pre * 0., label='zero')
    plt.xlabel('item')
    plt.ylabel('m')
    plt.title(seq_type)
    plt.legend()
    f_t = plt.gcf()
    mkdir(dir)
    f_t.savefig(os.path.join(dir, '{}_{}.png'.format(seq_type, metric)))
    f_t.clear()


def print_items(mpjpe_errors_list, labels_list, metric, file):
    mpjpe_errors = {}
    for i, key in enumerate(labels_list):
        print(f'len of {key} is {len(mpjpe_errors_list[i])}')
        if len(mpjpe_errors_list[i]) != 0:
            mpjpe_errors.update(
                {key: torch.stack(mpjpe_errors_list[i])}
            )
    print_metrics(mpjpe_errors, metric=metric, f=file)


def run_eval_and_show(config, val_dataloader_normal, val_dataloader_fast, EvRGBStereo_model, _loss):
    mano_layer = MANO(config['data']['smplx_path'], use_pca=False, is_rhand=True).cuda()
    render = Render(config)

    if config['exper']['distributed']:
        EvRGBStereo_model = torch.nn.parallel.DistributedDataParallel(
            EvRGBStereo_model, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )
        mano_layer = torch.nn.parallel.DistributedDataParallel(
            mano_layer, device_ids=[int(os.environ["LOCAL_RANK"])],
            output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True,
        )

    EvRGBStereo_model.eval()

    start_eval_time = time.time()
    end = time.time()

    batch_time = AverageMeter()
    infer_time = AverageMeter()

    labels_list = ['n_f', 'n_r', 'h_f', 'h_r', 'f_f', 'f_r', 'fast']
    colors_list = ['r', 'g', 'b', 'gold', 'purple', 'cyan', 'm']
    steps = config['exper']['preprocess']['steps']

    errors_list = []
    for i in range(4):
        errors_list.append([])
        for j in range(steps):
            errors_list[i].append([])
            for k in range(len(labels_list)):
                errors_list[i][j].append([])

    metrics = ['MPJPE', 'PA_MPJPE', 'MPVPE', 'PA_MPVPE']

    mkdir(config['exper']['output_dir'])

    file = open(os.path.join(config['exper']['output_dir'], 'error_joints.txt'), 'a')

    last_seq = 0

    with torch.no_grad():
        for iteration, (frames, meta_data) in enumerate(val_dataloader_normal):
            if last_seq != str(meta_data[0]['seq_id'][0].item()):
                last_seq = str(meta_data[0]['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            device = 'cuda'
            batch_size = frames[0]['rgb'].shape[0]
            frames = to_device(frames, device)
            meta_data = to_device(meta_data, device)
            t_start_infer = time.time()
            preds = EvRGBStereo_model(frames)
            infer_time.update(time.time() - t_start_infer, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if iteration == 0:
                # for flops
                flops = FlopCountAnalysis(EvRGBStereo_model, frames)
                print('FLOPs: {} G FLOPs'.format(flops.total() / batch_size / 1024**3))
                file.write('FLOPs: {} G FLOPs\n'.format(flops.total() / batch_size / 1024**3))

                # for parameters
                all_params = sum(p.numel() for p in EvRGBStereo_model.parameters()) / 1024.**2
                print('Params: {} M'.format(all_params))
                file.write('Params: {} M\n'.format(all_params))

                start = time.time()

            for step in range(steps):

                bbox_valid = meta_data[step]['bbox_valid']
                mano_valid = meta_data[step]['mano_valid'] * bbox_valid
                joints_3d_valid = meta_data[step]['joints_3d_valid'] * bbox_valid

                if step==0 and((~joints_3d_valid).any() or (~mano_valid).any()):
                    print('joints_3d_valid', joints_3d_valid)
                    print('mano_valid', mano_valid)
                    print('bbox_valid', bbox_valid)
                    print('seq_type  ', meta_data[step]['seq_type'])
                
                manos = meta_data[step]['mano']
                gt_dest_mano_output = mano_layer(
                    global_orient=manos['rot_pose'].reshape(-1, 3),
                    hand_pose=manos['hand_pose'].reshape(-1, 45),
                    betas=manos['shape'].reshape(-1, 10),
                    transl=manos['trans'].reshape(-1, 3)
                )
                gt_3d_joints = gt_dest_mano_output.joints
                gt_root = gt_3d_joints[:, 0, :]
                gt_3d_joints_sub = gt_3d_joints - gt_root[:, None, :]
                pred_3d_joints = preds[step][-1]['pred_3d_joints']
                pred_3d_joints_sub = pred_3d_joints - pred_3d_joints[:, :1]
                mpjpe_eachjoint_eachitem = torch.sqrt(
                    torch.sum((pred_3d_joints_sub - gt_3d_joints_sub) ** 2, dim=-1)) * 1000.
                
                update_errors_list(errors_list[0][step], mpjpe_eachjoint_eachitem[joints_3d_valid], meta_data[step]['seq_type'][joints_3d_valid])
                align_pred_3d_joints_sub = compute_similarity_transform_batch(pred_3d_joints_sub, gt_3d_joints_sub)
                pa_mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((align_pred_3d_joints_sub - gt_3d_joints_sub.detach().cpu()) ** 2, dim=-1)) * 1000.
                update_errors_list(errors_list[1][step], pa_mpjpe_eachjoint_eachitem[joints_3d_valid.cpu()], meta_data[step]['seq_type'][joints_3d_valid])

                gt_3d_joints_from_mesh = mano_layer.get_3d_joints(gt_dest_mano_output.vertices)
                gt_vertices_sub = gt_dest_mano_output.vertices - gt_3d_joints_from_mesh[:,:1,:]

                pred_3d_joints_from_mesh = mano_layer.get_3d_joints(preds[step][-1]['pred_vertices'])
                pred_vertices_sub = preds[step][-1]['pred_vertices'] - pred_3d_joints_from_mesh[:, :1, :]
                
                error_vertices = torch.sqrt(torch.sum((pred_vertices_sub - gt_vertices_sub) ** 2, dim=-1)) * 1000.
                #error_vertices = torch.mean(torch.abs(preds[step][-1]['pred_vertices'] - gt_vertices_sub), dim=(1, 2))
                update_errors_list(errors_list[2][step], error_vertices[mano_valid], meta_data[step]['seq_type'][mano_valid])
                
                aligned_pred_vertices = compute_similarity_transform_batch(pred_vertices_sub, gt_vertices_sub)
                #pa_error_vertices = torch.mean(torch.abs(aligned_pred_vertices - gt_vertices_sub.detach().cpu()), dim=(1, 2))
                pa_error_vertices = torch.sqrt(torch.sum((aligned_pred_vertices - gt_vertices_sub.detach().cpu()) ** 2, dim=-1)) * 1000.
                update_errors_list(errors_list[3][step], pa_error_vertices[mano_valid.cpu()], meta_data[step]['seq_type'][mano_valid])


            if config['eval']['output']['save']:
                # print('step', steps)
                for step in range(steps):
                    segments = len(preds[step])
                    # print('segments', segments)
                    for seg in range(segments):
                        if step >=1:
                            wrist_3d_inter = ((seg + 1) * meta_data[step]['3d_joints'][:, :1] + (segments - 1 - seg) *
                                              meta_data[step - 1]['3d_joints'][:, :1]) / segments
                        else:
                            wrist_3d_inter = meta_data[step]['3d_joints'][:, :1]
                        pred_3d_joints_from_mesh = mano_layer.get_3d_joints(preds[step][-1]['pred_vertices'])
                        predicted_meshes = preds[step][seg]['pred_vertices'] - pred_3d_joints_from_mesh[:,:1,:] + wrist_3d_inter
                        if config['eval']['output']['mesh']:
                            for i in range(batch_size):
                                seq_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()))
                                mkdir(seq_dir)
                                if config['eval']['output']['mesh']:
                                    mesh_dir = op.join(seq_dir, 'mesh')
                                    mkdir(mesh_dir)
                                    with open(os.path.join(mesh_dir, 'annot_{}_step_{}.obj'.format(meta_data[step]['annot_id'][i].item(), step)), 'w') as file_object:
                                        for ver in predicted_meshes[i].detach().cpu().numpy():
                                            print('v %f %f %f'%(ver[0], ver[1], ver[2]), file=file_object)
                                        faces = mano_layer.faces
                                        for face in faces:
                                            print('f %d %d %d'%(face[0]+1, face[1]+1, face[2]+1), file=file_object)
                        if config['eval']['output']['rendered']:
                            key = config['eval']['output']['vis_rendered']
                            if key == 'rgb':
                                img_bg = frames[step][key+'_ori']
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]
                            else:
                                img_bg = frames[step][key + '_ori'][seg]
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]

                            img_render = render.visualize(
                                K=meta_data[step]['K_'+key].detach(),
                                R=meta_data[step]['R_'+key].detach(),
                                t=meta_data[step]['t_'+key].detach(),
                                hw=hw,
                                img_bg=img_bg.cpu(),
                                vertices=predicted_meshes.detach(),
                            )
                            for i in range(img_render.shape[0]):
                                img_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()), 'rendered')
                                mkdir(img_dir)
                                imageio.imwrite(op.join(img_dir, 'annot_{}_step_{}_seg_{}.jpg'.format(meta_data[0]['annot_id'][i].item(), step, seg)),
                                                (img_render[i].detach().cpu().numpy() * 255).astype(np.uint8))

                        if config['eval']['output']['vis_rgb']:
                            key = 'rgb'
                            if key == 'rgb':
                                img_bg = frames[step][key+'_ori']
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]
                            else:
                                img_bg = frames[step][key + '_ori'][seg]
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]

                            for i in range(img_bg.shape[0]):
                                img_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()), 'vis_rgb')
                                mkdir(img_dir)
                                imageio.imwrite(op.join(img_dir, 'annot_{}_step_{}_seg_{}.jpg'.format(meta_data[0]['annot_id'][i].item(), step, seg)),
                                                (img_bg[i].detach().cpu().numpy() * 255).astype(np.uint8))
                            img_render = render.visualize(
                                K=meta_data[step]['K_'+key].detach(),
                                R=meta_data[step]['R_'+key].detach(),
                                t=meta_data[step]['t_'+key].detach(),
                                hw=hw,
                                img_bg=img_bg.cpu(),
                                vertices=predicted_meshes.detach(),
                            )


            if (iteration - 1) % config['utils']['logging_steps'] == 0.5:
                eta_seconds = batch_time.avg * (len(val_dataloader_normal) / config['exper']['per_gpu_batch_size'] - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if is_main_process():
                    print(
                        ' '.join(
                            ['eta: {eta}', 'iter: {iter}', 'max mem : {memory:.0f}', ]
                        ).format(eta=eta_string, iter=iteration,
                                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                        + '   compute: {:.4f}'.format(batch_time.avg)
                    )

        '''
        for fast sequences
        '''
        for iteration, (frames, meta_data) in enumerate(val_dataloader_fast):

            if last_seq != str(meta_data[0]['seq_id'][0].item()):
                last_seq = str(meta_data[0]['seq_id'][0].item())
                print('Now for seq id: ', last_seq)

            device = 'cuda'
            batch_size = frames[0]['rgb'].shape[0]
            frames = to_device(frames, device)
            meta_data = to_device(meta_data, device)

            preds = EvRGBStereo_model(frames, return_att=True, decode_all=True)
            batch_time.update(time.time() - end)
            end = time.time()
            steps = len(preds)
            for step in range(steps):
                joints_2d_valid = meta_data[step]['joints_2d_valid_ev'] * meta_data[step]['bbox_valid']

                pred_3d_joints = preds[step][-1]['pred_3d_joints']
                pred_3d_joints_abs = pred_3d_joints + meta_data[step]['3d_joints'][:, :1]
                pred_3d_joints_abs = torch.bmm(meta_data[step]['R_event'].reshape(-1, 3, 3), pred_3d_joints_abs.transpose(2, 1)).transpose(2, 1) + meta_data[step]['t_event'].reshape(-1, 1, 3)
                # todo check here!!
                pred_2d_joints = torch.bmm(meta_data[step]['K_event'], pred_3d_joints_abs.permute(0, 2, 1)).permute(0, 2, 1)
                pred_2d_joints = pred_2d_joints[:, :, :2] / pred_2d_joints[:, :, 2:]
                gt_2d_joints = meta_data[step]['2d_joints_event']
                pred_2d_joints_aligned = pred_2d_joints - pred_2d_joints[:, :1] + gt_2d_joints[:, :1]
        
                mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_2d_joints_aligned - gt_2d_joints) ** 2, dim=-1))
                
                update_errors_list(errors_list[0][step], mpjpe_eachjoint_eachitem[joints_2d_valid], meta_data[step]['seq_type'][joints_2d_valid])
            if config['eval']['output']['save']:
                for step in range(steps):
                    segments = len(preds[step])
                    for seg in range(segments):
                
                        wrist_3d_inter = ((seg+1)*meta_data[step]['3d_joints'][:, :1]+(segments-1-seg)*meta_data[step]['meta_data_prev']['3d_joints'][:, :1])/segments
                        pred_3d_joints_from_mesh = mano_layer.get_3d_joints(preds[step][seg]['pred_vertices'])
                        predicted_meshes = preds[step][seg]['pred_vertices'] - pred_3d_joints_from_mesh[:,:1,:] + wrist_3d_inter
                        if config['eval']['output']['mesh']:
                            for i in range(batch_size):
                                seq_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()))
                                mkdir(seq_dir)
                                if config['eval']['output']['mesh']:
                                    mesh_dir = op.join(seq_dir, 'mesh')
                                    mkdir(mesh_dir)
                                    with open(os.path.join(mesh_dir, 'annot_{}_step_{}_seg_{}.obj'.format(meta_data[step]['annot_id'][i].item(), step, seg)), 'w') as file_object:
                                        for ver in predicted_meshes[i].detach().cpu().numpy():
                                            print('v %f %f %f'%(ver[0], ver[1], ver[2]), file=file_object)
                                        faces = mano_layer.faces
                                        for face in faces:
                                            print('f %d %d %d'%(face[0]+1, face[1]+1, face[2]+1), file=file_object)
                        if config['eval']['output']['rendered']:
                            key = config['eval']['output']['vis_rendered']
                            if key == 'rgb':
                                img_bg = frames[step][key+'_ori']
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]
                            else:
                                img_bg = frames[step][key + '_ori'][seg]
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]

                            img_render = render.visualize(
                                K=meta_data[step]['K_'+key].detach(),
                                R=meta_data[step]['R_'+key].detach(),
                                t=meta_data[step]['t_'+key].detach(),
                                hw=hw,
                                img_bg=img_bg.cpu(),
                                vertices=predicted_meshes.detach(),
                            )
                            for i in range(img_render.shape[0]):
                                img_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()), 'rendered')
                                mkdir(img_dir)
                                imageio.imwrite(op.join(img_dir, 'annot_{}_step_{}_seg_{}.jpg'.format(meta_data[0]['annot_id'][i].item(), step, seg)),
                                                (img_render[i].detach().cpu().numpy() * 255).astype(np.uint8))
                        
                        if config['eval']['output']['vis_event']:
                            key = 'event'
                            if key == 'rgb':
                                img_bg = frames[step][key+'_ori']
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]
                            else:
                                img_bg = frames[step][key + '_ori'][seg]
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]

                            for i in range(img_render.shape[0]):
                                img_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()), 'vis_event')
                                mkdir(img_dir)
                                imageio.imwrite(op.join(img_dir, 'annot_{}_step_{}_seg_{}.jpg'.format(meta_data[0]['annot_id'][i].item(), step, seg)),
                                                (img_bg[i].detach().cpu().numpy() * 255).astype(np.uint8))
                        
                        if config['eval']['output']['vis_rgb']:
                            key = 'rgb'
                            if key == 'rgb':
                                img_bg = frames[step][key+'_ori']
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]
                            else:
                                img_bg = frames[step][key + '_ori'][seg]
                                _, h, w, _ = img_bg.shape
                                hw = [h, w]

                            for i in range(img_render.shape[0]):
                                img_dir = op.join(config['exper']['output_dir'], str(meta_data[step]['seq_id'][i].item()), 'vis_rgb')
                                mkdir(img_dir)
                                imageio.imwrite(op.join(img_dir, 'annot_{}_step_{}_seg_{}.jpg'.format(meta_data[0]['annot_id'][i].item(), step, seg)),
                                                (img_bg[i].detach().cpu().numpy() * 255).astype(np.uint8))

    for step in range(config['exper']['preprocess']['steps']):
        file.write('Step: {}'.format(step) + '\n')
        for i, metric in enumerate(metrics):
            print_items(errors_list[i][step], labels_list, metric, file)
            file.write('\n')
        file.write('\n\n')

    for step in range(config['exper']['preprocess']['steps']):
        for i, key in enumerate(labels_list):
            if len(errors_list[0][step][i]) != 0:
                errors_seq = torch.stack(errors_list[0][step][i])
                print_sequence_error(errors_seq, metrics[0], labels_list[i]+'_step'+str(step), os.path.join(config['exper']['output_dir'], 'seq_errors'))
                if config['eval']['output']['errors']:
                    error_dir = op.join(config['exper']['output_dir'], 'errors', labels_list[i]+'_step'+str(step))
                    mkdir(error_dir)
                    torch.save(errors_seq, os.path.join(error_dir, 'kps_errors.pt'))

    # if config['eval']['output']['save']:
    #     if config['eval']['output']['errors']:
    #         error_dir = op.join(config['exper']['output_dir'], 'error')
    #         mkdir(error_dir)
    #         torch.save(l_mpjpe_errors, os.path.join(error_dir, 'l_mpjpe_errors.pt'))
    #         torch.save(l_vertices_errors, os.path.join(error_dir, 'l_vertices_errors.pt'))
    #         torch.save(l_pa_mpjpe_errors, os.path.join(error_dir, 'l_pa_mpjpe_errors.pt'))
    #         torch.save(l_pa_vertices_errors, os.path.join(error_dir, 'l_pa_vertices_errors.pt'))


    print('Inference time each item: {}'.format(infer_time.avg))
    file.write('Inference time each item: {}\n'.format(infer_time.avg))
    file.close()
    total_eval_time = time.time() - start_eval_time
    total_time_str = str(datetime.timedelta(seconds=total_eval_time))
    print('Total eval time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_eval_time / ((len(val_dataloader_normal) + len(val_dataloader_fast)) / config['exper']['per_gpu_batch_size']))
    )

    return


def get_config():
    #warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--config', type=str, default='src/configs/config/evrgbhand.yaml')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--config_merge', type=str, default='')
    parser.add_argument('--run_eval_only', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--s', default=1.0, type=float, help='scale')
    parser.add_argument('--r', default=0., type=float, help='rotate')
    parser.add_argument('--output_dir', type=str,
                        default='')
    args = parser.parse_args()
    config = ConfigParser(args.config)

    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config
    if args.output_dir != '':
        config['exper']['output_dir'] = args.output_dir
    if args.resume_checkpoint != '':
        config['exper']['resume_checkpoint'] = args.resume_checkpoint
    config['exper']['run_eval_only'] = args.run_eval_only
    dataset_config = ConfigParser(config['data']['dataset_yaml']).config
    config['data']['dataset_info'] = dataset_config
    if 'augment' not in config['eval'].keys():
        config['eval']['augment'] = {}
    config['eval']['augment']['scale'] = args.s
    config['eval']['augment']['rot'] = args.r


    return config


def main(config):
    global tf_logger
    # Setup CUDA, GPU & distributed training
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(config['exper']['num_workers'])
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    config['exper']['distributed'] = num_gpus > 1
    config['exper']['device'] = torch.device(config['exper']['device'])
    if config['exper']['distributed']:
        print("Init distributed training on local rank {}".format(int(os.environ["LOCAL_RANK"])))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            backend='nccl'  # , init_method='env://'
        )
        synchronize()
        setup_for_distributed(int(os.environ["LOCAL_RANK"]) == 0)
    else:
        print("Not using distributed mode")
        
    mkdir(config['exper']['output_dir'])
  
    if is_main_process() and not config['exper']['run_eval_only']:
        mkdir(os.path.join(config['exper']['output_dir'], 'tf_logs'))
        tf_logger = SummaryWriter(os.path.join(config['exper']['output_dir'], 'tf_logs'))
    else:
        tf_logger = None

    set_seed(config['exper']['seed'], num_gpus)
    print(f'seed: {config["exper"]["seed"]}')
    print("Using {} GPUs".format(num_gpus))

    start_iter = 0

    _model = build_model(config)
    if config['exper']['run_eval_only'] == True or (config['exper']['resume_checkpoint'] != None and\
            config['exper']['resume_checkpoint'] != 'None'):

        #_model = EvRGBStereo(config=config)
        checkpoint = torch.load(config['exper']['resume_checkpoint'], map_location=torch.device('cpu'))

        missing_keys, unexpected_keys = _model.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    _model.to(config['exper']['device'])
    
    _loss = Loss(config)
    _loss.to(config['exper']['device'])

    if config['exper']['run_eval_only'] == True:
        val_dataloader_normal = make_hand_data_loader(config)
        config_fast = copy.deepcopy(config)
        config_fast['eval']['fast'] = True
        val_dataloader_fast = make_hand_data_loader(config_fast)
        run_eval_and_show(config, val_dataloader_normal, val_dataloader_fast, _model, _loss)

    else:
        train_dataloader = make_hand_data_loader(config, start_iter)
        run(config, train_dataloader, _model, _loss)

    if is_main_process() and not config['exper']['run_eval_only']:
        tf_logger.close()


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    config = get_config()
    main(config)
