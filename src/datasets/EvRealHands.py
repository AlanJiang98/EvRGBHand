import cv2
import os
import os.path as osp
import numpy as np
import torch
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import multiprocessing as mp
import matplotlib.pyplot as plt
import roma
from src.utils.dataset_utils import json_read, extract_data_from_aedat4, undistortion_points, event_representations
from src.utils.joint_indices import indices_change
#import albumentations as A
from src.configs.config_parser import ConfigParser
from smplx import MANO
from src.utils.augment import EvRGBDegrader
from src.utils.comm import is_main_process
from scipy.interpolate import interp1d


from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection


def collect_data(data_seq):
    global data_seqs
    if not data_seq:
        return
    for key in data_seq.keys():
        data_seqs[key] = data_seq[key]


class EvRealHands(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = {}
        self.data_config = None
        self.load_events_annotations()
        self.process_samples()
        self.mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True)
        self.rgb_degrader = EvRGBDegrader(self.config['exper']['augment']['rgb_photometry'], not self.config['exper']['run_eval_only'])
        self.event_degrader = EvRGBDegrader(self.config['exper']['augment']['event_photometry'], not self.config['exper']['run_eval_only'])
        self.get_bbox_inter_f()
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
        if is_main_process():
            print('EvRealHands length is ', self.__len__())

    @staticmethod
    def get_events_annotations_per_sequence(dir, is_train=True, is_fast=False):
        if not osp.isdir(dir):
            raise FileNotFoundError('illegal directions for event sequence: {}'.format(dir))
        id = dir.split('/')[-1]
        annot = json_read(os.path.join(dir, 'annot.json'))

        if is_train and not (annot['motion_type'] == 'fast'):
            mano_ids = [int(id) for id in annot['manos'].keys()]
            if annot['scene'] != 'normal':
                return {}
        elif not is_train:
            if is_fast:
                if annot['motion_type'] == 'fast':
                    mano_ids = [int(id) for id in annot['3d_joints'].keys()]
                else:
                    return {}
            else:
                if not (annot['motion_type'] == 'fast') and annot['annoted']:
                    mano_ids = [int(id) for id in annot['manos'].keys()]
                else:
                    return {}
        else:
            return {}
        mano_ids.sort()

        # mano_ids_np = np.array(mano_ids, dtype=np.int32)
        # indices = np.diff(mano_ids_np) == 1
        # annot['sample_ids'] = [str(id) for id in mano_ids_np[1:][indices]]
        # deal with static hand pose
        #todo
        # mano_ids = mano_ids[15:-11]
        mano_ids = mano_ids[20:-11]
        annot['sample_ids'] = [str(id) for id in mano_ids]
        K_old = np.array(annot['camera_info']['event']['K_old'])
        K_new = np.array(annot['camera_info']['event']['K_new'])
        dist = np.array(annot['camera_info']['event']['dist'])
        undistortion_param = [K_old, dist, K_new]
        events, _, _ = extract_data_from_aedat4(osp.join(dir, 'event.aedat4'), is_event=True)
        events = np.vstack([events['x'], events['y'], events['polarity'], events['timestamp']]).T
        if not np.all(np.diff(events[:, 3]) >= 0):
            events = events[np.argsort(events[:, 3])]
        first_event_time = events[0, 3]
        events[:, 3] = events[:, 3] - first_event_time
        events = events.astype(np.float32)
        # undistortion the events
        events[:, :2], legal_indices = undistortion_points(events[:, :2], undistortion_param[0],
                                                                undistortion_param[1],
                                                                undistortion_param[2],
                                                                set_bound=True, width=346,
                                                                height=260)
        events = events[legal_indices]
        data = {}
        data[id] = {
            'event': events,
            'annot': annot,
        }
        if is_main_process():
            print('Seq {} is over!'.format(dir))
        return data

    def load_events_annotations(self):
        if self.config['exper']['run_eval_only']:
            data_config = ConfigParser(self.config['data']['dataset_info']['evrealhands']['eval_yaml'])
        else:
            data_config = ConfigParser(self.config['data']['dataset_info']['evrealhands']['train_yaml'])
        self.data_config = data_config.config
        all_sub_2_seq_ids = json_read(
            osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], self.data_config['seq_ids_path'])
        )
        self.seq_ids = []
        for sub_id in self.data_config['sub_ids']:
            self.seq_ids += all_sub_2_seq_ids[str(sub_id)]
        
        if self.config['exper']['debug']:
            if self.config['exper']['run_eval_only']:
                # todo change!
                print('debug mode for evrealhands')
                seq_ids = ['53','27'] #''1', '53', 52', '20', '21', '26', '27', '25']
            else:
                seq_ids = ['1', '24', '18', '26']#['24', '18']
            for seq_id in seq_ids:
                data = self.get_events_annotations_per_sequence(osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id),
                                                            not self.config['exper']['run_eval_only'], self.config['eval']['fast'])
                self.data.update(data)
            self.seq_ids = seq_ids
        else:
            global data_seqs
            data_seqs = {}
            pool = mp.Pool(mp.cpu_count())
            for seq_id in self.seq_ids:
                pool.apply_async(EvRealHands.get_events_annotations_per_sequence,
                                 args=(osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id),
                                       not self.config['exper']['run_eval_only'], self.config['eval']['fast'],),
                                 callback=collect_data)
            pool.close()
            pool.join()
            self.data = data_seqs

        if is_main_process():
            print('All the sequences for EvRealHands: number: {} items: {}'.format(len(self.data.keys()), self.data.keys()))

    def process_samples(self):
        '''
        get samples for the dataset
        '''
        self.sample_info = {
            'seq_ids': [],
            'cam_pair_ids': [],
            'samples_per_seq': None,
            'samples_sum': None
        }
        samples_per_seq = []

        for seq_id in self.data.keys():
            for cam_pair_id, cam_pair in enumerate(self.data_config['camera_pairs']):
                samples_per_seq.append(len(self.data[seq_id]['annot']['sample_ids']))
                self.sample_info['seq_ids'].append(seq_id)
                self.sample_info['cam_pair_ids'].append(cam_pair_id)
        self.sample_info['samples_per_seq'] = np.array(samples_per_seq, dtype=np.int32)
        self.sample_info['samples_sum'] = np.cumsum(self.sample_info['samples_per_seq'])


    def get_info_from_sample_id(self, sample_id):
        '''
        get image_id, event time stamp, annot id from sample id
        '''
        sample_info_index = np.sum(self.sample_info['samples_sum'] <= sample_id, dtype=np.int32)
        seq_id = self.sample_info['seq_ids'][sample_info_index]
        cam_pair = self.data_config['camera_pairs'][self.sample_info['cam_pair_ids'][sample_info_index]]
        if sample_info_index == 0:
            seq_loc = sample_id
        else:
            seq_loc = sample_id - self.sample_info['samples_sum'][sample_info_index-1]

        annot_id = self.data[seq_id]['annot']['sample_ids'][seq_loc]
        return seq_id, cam_pair, annot_id

    def load_img(self, path, order='RGB'):
        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)
        if order == 'RGB':
            img = img[:, :, ::-1].copy()

        img = img.astype(np.float32)
        return img

    def get_annotations(self, seq_id, cam_pair, annot_id):
        meta_data = {}
        meta_data['delta_time'] = self.data[seq_id]['annot']['delta_time']
        camera_pair_0 = self.data[seq_id]['annot']['camera_info'][cam_pair[0]]
        meta_data['R_event'] = torch.tensor(camera_pair_0['R'], dtype=torch.float32).view(3, 3)
        meta_data['K_event'] = torch.tensor(camera_pair_0['K'], dtype=torch.float32).view(3, 3)
        meta_data['t_event'] = torch.tensor(camera_pair_0['T'], dtype=torch.float32).view(3) / 1000.
        camera_pair_1 = self.data[seq_id]['annot']['camera_info'][cam_pair[1]]
        meta_data['R_rgb'] = torch.tensor(camera_pair_1['R'], dtype=torch.float32).view(3, 3)
        meta_data['K_rgb'] = torch.tensor(camera_pair_1['K'], dtype=torch.float32).view(3, 3)
        meta_data['t_rgb'] = torch.tensor(camera_pair_1['T'], dtype=torch.float32).view(3) / 1000.

        if annot_id % 1 != 0:
            annot_id = '-1'
        else:
            annot_id = str(int(annot_id))

        if self.data[seq_id]['annot']['motion_type'] == 'fast' and\
                annot_id in self.data[seq_id]['annot']['2d_joints'][cam_pair[0]].keys() and\
                self.data[seq_id]['annot']['2d_joints'][cam_pair[0]][annot_id] != []:
            meta_data['2d_joints_event'] = torch.tensor(self.data[seq_id]['annot']['2d_joints'][cam_pair[0]][annot_id], dtype=torch.float32).view(21, 2)[indices_change(2, 1)]
            joints_2d_valid_ev = True
        else:
            meta_data['2d_joints_event'] = torch.zeros((21, 2), dtype=torch.float32)
            joints_2d_valid_ev = False

        if annot_id not in self.data[seq_id]['annot']['2d_joints'][cam_pair[1]].keys() or \
                self.data[seq_id]['annot']['2d_joints'][cam_pair[1]][annot_id] == []:
            meta_data['2d_joints_rgb'] = torch.zeros((21, 2), dtype=torch.float32)
            joints_2d_valid_rgb = False
        else:
            meta_data['2d_joints_rgb'] = torch.tensor(self.data[seq_id]['annot']['2d_joints'][cam_pair[1]][annot_id], dtype=torch.float32).view(21, 2)[indices_change(2, 1)]
            joints_2d_valid_rgb = True

        if annot_id in self.data[seq_id]['annot']['3d_joints'].keys():
            meta_data['3d_joints'] = torch.tensor(self.data[seq_id]['annot']['3d_joints'][annot_id], dtype=torch.float32).view(-1, 3)[
                             indices_change(2, 1)] / 1000.
            joints_3d_valid = True
        else:
            meta_data['3d_joints'] = torch.zeros((21, 3), dtype=torch.float32)
            joints_3d_valid = False

        if annot_id in self.data[seq_id]['annot']['manos'].keys():
            mano = self.data[seq_id]['annot']['manos'][annot_id]
            meta_data['mano'] = {}
            meta_data['mano']['rot_pose'] = torch.tensor(mano['rot'], dtype=torch.float32).view(-1)
            meta_data['mano']['hand_pose'] = torch.tensor(mano['hand_pose'], dtype=torch.float32).view(-1)
            meta_data['mano']['shape'] = torch.tensor(mano['shape'], dtype=torch.float32).view(-1)
            meta_data['mano']['trans'] = torch.tensor(mano['trans'], dtype=torch.float32).view(3)
            meta_data['mano']['root_joint'] = torch.tensor(mano['root_joint'], dtype=torch.float32).view(3)
            mano_valid = True
        else:
            meta_data['mano'] = {}
            meta_data['mano']['rot_pose'] = torch.zeros((3, ), dtype=torch.float32)
            meta_data['mano']['hand_pose'] = torch.zeros((45, ), dtype=torch.float32)
            meta_data['mano']['shape'] = torch.zeros((10, ), dtype=torch.float32)
            meta_data['mano']['trans'] = torch.zeros((3, ), dtype=torch.float32)
            meta_data['mano']['root_joint'] = torch.zeros((3, ), dtype=torch.float32)
            mano_valid = False

        meta_data['joints_2d_valid_ev'] = joints_2d_valid_ev
        meta_data['joints_2d_valid_rgb'] = joints_2d_valid_rgb
        meta_data['joints_3d_valid'] = joints_3d_valid
        meta_data['mano_valid'] = mano_valid
        return meta_data

    def get_event_repre(self, seq_id, indices):
        event = self.data[seq_id]['event']
        if indices[0] == 0 or indices[-1] >= event.shape[0] - 1 or indices[-1]==indices[0]:
            return torch.zeros((260, 346, 3))
        else:
            tmp_events = event[indices[0]:indices[1]].copy()
            tmp_events[:, 3] = (tmp_events[:, 3] - event[indices[0], 3]) / (event[indices[1], 3] - event[indices[0], 3])
            tmp_events = torch.tensor(tmp_events, dtype=torch.float32)
            ev_frame = event_representations(tmp_events, repre=self.config['exper']['preprocess']['ev_repre'], hw=(260, 346))
            if ev_frame.shape[0] == 2:
                ev_frame = torch.cat([ev_frame, torch.zeros((1, 260, 346))], dim=0)
            #assert ev_frame.shape[0] == 2, seq_id
            return ev_frame.permute(1, 2, 0)

    def get_augment_param(self):
        # augmentation for event and rgb
        augs = [[], []]
        if not self.config['exper']['run_eval_only']:
            scale_range, trans_range, rot_range = self.config['exper']['augment']['geometry']['scale'], \
                                                  self.config['exper']['augment']['geometry']['trans'], \
                                                  self.config['exper']['augment']['geometry']['rot']
            for i in range(2):
                scale = min(2*scale_range, max(-2*scale_range, np.random.randn() * scale_range)) + 1
                trans_x = min(2*trans_range, max(-2*trans_range, np.random.randn() * trans_range))
                trans_y = min(2 * trans_range, max(-2 * trans_range, np.random.randn() * trans_range))
                rot = min(2*rot_range, max(-2*rot_range, np.random.randn() * rot_range))
                augs[i] += [trans_x, trans_y, rot, scale]
        else:
            for i in range(2):
                augs[i] += [0, 0, 0, 1]
        return augs

    def get_transform(self, R, t):
        tf = torch.eye(4)
        tf[:3, :3] = R
        tf[:3, 3] = t
        return tf

    def get_trans_from_augment(self, augs, meta_data, H, W, view='rgb'):
        mano_key = 'mano'
        K = meta_data['K_'+view]
        R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
        t_wm = meta_data[mano_key]['trans']

        tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
        tf_cw = self.get_transform(meta_data['R_'+view], meta_data['t_'+view])

        tf_cm = tf_cw @ tf_wm

        # trans augmentation
        aff_trans_2d = torch.eye(3)
        # trans_x_2d = augs[0] * W
        # trans_y_2d = augs[1] * H
        # aff_trans_2d[0, 2] = trans_x_2d
        # aff_trans_2d[1, 2] = trans_y_2d
        tf_trans_3d = torch.eye(4)
        # trans_x_3d = trans_x_2d * tf_cm[2, 3] / meta_data['K_'+view][0, 0]
        # trans_y_3d = trans_y_2d * tf_cm[2, 3] / meta_data['K_'+view][1, 1]
        # tf_trans_3d[0, 3] = trans_x_3d
        # tf_trans_3d[1, 3] = trans_y_3d

        # rotation
        rot_2d = roma.rotvec_to_rotmat(augs[2] * torch.tensor([0.0, 0.0, 1.0]))
        tf_rot_3d = self.get_transform(rot_2d, torch.zeros(3))
        aff_rot_2d = torch.eye(3)
        aff_rot_2d[:2, :2] = rot_2d[:2, :2]
        aff_rot_2d[0, 2] = (1 - rot_2d[0, 0]) * K[0, 2] - rot_2d[0, 1] * K[1, 2]
        aff_rot_2d[1, 2] = (1 - rot_2d[1, 1]) * K[1, 2] - rot_2d[1, 0] * K[0, 2]

        tf_3d_tmp = tf_rot_3d @ tf_trans_3d @ tf_cm
        # scale
        tf_scale_3d = torch.eye(4)
        # tf_scale_3d[2, 2] = 1.0 / augs[3]
        #tf_scale_3d[2, 3] = (1.0 / augs[3] - 1.) * tf_3d_tmp[2, 3]
        aff_scale_2d = torch.eye(3)
        # aff_scale_2d[0, 0], aff_scale_2d[1, 1] = augs[3], augs[3]
        # aff_scale_2d[:2, 2] = (1-augs[3]) * K[:2, 2]

        aff_2d_final = (aff_scale_2d @ aff_rot_2d @ aff_trans_2d)[:2, :]
        # frame_aug = cv2.warpAffine(np.array(frame), aff_2d_final.numpy(), (W, H), flags=cv2.INTER_LINEAR)

        tf_3d_final = tf_scale_3d @ tf_rot_3d @ tf_trans_3d @ tf_cm
        tf_cw = tf_3d_final @ tf_wm.inverse()

        # meta_data['R_'+view] = tf_cw[:3, :3]
        # meta_data['t_'+view] = tf_cw[:3, 3]

        return aff_2d_final.numpy(), tf_cw

    def get_bbox_from_joints(self, K, R, t, joints, rate=1.5):
        kps = (K @ (R @ joints.transpose(0, 1) + t.reshape(3, 1))).transpose(0, 1)
        kps = kps[:, :2] / kps[:, 2:]
        x_min = torch.min(kps[:, 0])
        x_max = torch.max(kps[:, 0])
        y_min = torch.min(kps[:, 1])
        y_max = torch.max(kps[:, 1])
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bbox_size = torch.max(torch.tensor([x_max-x_min, y_max-y_min])) * rate

        return torch.tensor([center_x, center_y, bbox_size]).int()

    def valid_bbox(self, bbox, hw=[260, 346]):
        if (bbox[:2] < 0).any() or bbox[0] > hw[1] or bbox[1] > hw[0] or bbox[2] < 10 or bbox[2] > 400:
            return False
        else:
            return True

    def get_default_bbox(self, hw=[260, 346], size=128):
        return torch.tensor([hw[1] / 2, hw[0] / 2, size], dtype=torch.float32)

    def crop(self, bbox, frame, size, hw=[260, 346]):
        lf_top = (bbox[:2] - bbox[2] / 2).int()
        rt_dom = lf_top + bbox[2].int()
        if lf_top[0] < 0 or lf_top[1] < 0 or rt_dom[0] > hw[1] or rt_dom[1] > hw[0]:
            frame = cv2.copyMakeBorder(frame, - min(0, lf_top[1].item()), max(rt_dom[1].item() - hw[0], 0),
                            -min(0, lf_top[0].item()), max(rt_dom[0].item() - hw[1], 0), cv2.BORDER_CONSTANT, value=0)# cv2.BORDER_REPLICATE) todo
            rt_dom[1] += -min(0, lf_top[1])
            lf_top[1] += -min(0, lf_top[1])
            rt_dom[0] += -min(0, lf_top[0])
            lf_top[0] += -min(0, lf_top[0])
        frame_crop = frame[lf_top[1]: rt_dom[1], lf_top[0]:rt_dom[0]]
        scale = size / bbox[2].float()
        frame_resize = cv2.resize(np.array(frame_crop), (size, size), interpolation=cv2.INTER_AREA)
        return lf_top, scale, frame_resize

    def plotshow(self, img):
        plt.imshow(img)
        plt.show()

    def get_l_event_indices(self, r_time, events, bbox = None):
        if self.config['exper']['preprocess']['event_range'] == 'time':
            l_win_range = [torch.log10(torch.tensor(x)) for x in self.config['exper']['preprocess']['left_window']]
            l_win = 10 ** (torch.rand(1)[0] * (l_win_range[1] - l_win_range[0]) + l_win_range[0])
            l_t = r_time - l_win
            indices = np.searchsorted(
                events[:, 3],
                # self.data[seq_id]['event'][:, 3],
                np.array([l_t, r_time])
            )
            index_l, index_r = indices[0], indices[1]
        elif self.config['exper']['preprocess']['event_range'] == 'num':
            timestamp = np.array(r_time, dtype=np.float32)
            index_r = np.searchsorted(
                events[:, 3],
                # self.data[seq_id]['event'][:, 3],
                timestamp
            )
            if not self.config['exper']['run_eval_only']:
                num = (np.random.rand(1) - 0.5) * 2 * self.config['exper']['preprocess']['num_var'] + self.config['exper']['preprocess']['num_window']
            else:
                num = self.config['exper']['preprocess']['num_window']
            index_l = max(0, index_r - int(num))
        elif self.config['exper']['preprocess']['event_range'] == 'bbox':
            bbox = bbox.numpy()
            lf_top = (bbox[:2] - bbox[2] / 2)
            rt_dom = lf_top + bbox[2]
            bbox_size = bbox[2].item()**2
            target_num = int(bbox_size * self.config['exper']['preprocess']['event_rate'])
            events_crop = events[(events[:, 0] > lf_top[0]) & (events[:, 0] < rt_dom[0]) & (events[:, 1] > lf_top[1]) & (events[:, 1] < rt_dom[1])]
            index_r_crop = events_crop.shape[0]
            index_r = events.shape[0]

            if events_crop.shape[0] == 0:
                return 0, index_r
            else:
                index_l_crop = max(0, index_r_crop - target_num)
                target_time = events_crop[index_l_crop, 3]
                index_l = np.searchsorted(
                    events[:, 3],
                    target_time
                )
        else:
            raise NotImplementedError('event range not implemented')
        return index_l, index_r

    def get_indices_from_timestamps(self, timestamps, seq_id, bbox = None):
        ## note to remove
        dt = timestamps[1] - timestamps[0]
        timestamps[0] = timestamps[0] - dt
        ####
        timestamps = np.array(timestamps, dtype=np.float32)
        event = self.data[seq_id]['event']
        indices = np.searchsorted(
            event[:, 3],
            timestamps
        )
        index_l, index_r = self.get_l_event_indices(timestamps[1], event[indices[0]:indices[1]].copy(), bbox=bbox)
        # todo crazy bug here!
        return index_l+indices[0], index_r+indices[0]

    def get_event_bbox_matrix_for_fast_sequences(self):
        for seq_id in self.data.keys():
            if self.data[seq_id]['annot']['motion_type'] == 'fast':
                bbox_ids = []
                for key in self.data[seq_id]['annot']['bbox']['event'].keys():
                    if self.data[seq_id]['annot']['bbox']['event'][key] != []:
                        bbox_ids.append(int(key))
                bbox_ids.sort()
                bbox_seq = np.zeros((len(bbox_ids), 4), dtype=np.float32)
                bbox_seq[:, 0] = np.array(bbox_ids, dtype=np.float32) * 1000000. / 15 + self.data[seq_id]['annot']['delta_time']
                # if the bbox is manual annotation, use it as GT bbox; else use machine annotation bbox
                K_old = np.array(self.data[seq_id]['annot']['camera_info']['event']['K_old'])
                K_new = np.array(self.data[seq_id]['annot']['camera_info']['event']['K_new'])
                dist = np.array(self.data[seq_id]['annot']['camera_info']['event']['dist'])
                undistortion_param = [K_old, dist, K_new]
                for i, id in enumerate(bbox_ids):
                    if str(id) in self.data[seq_id]['annot']['2d_joints']['event'].keys() and \
                            self.data[seq_id]['annot']['2d_joints']['event'][str(id)] != []:
                        kps2d = np.array(self.data[seq_id]['annot']['2d_joints']['event'][str(id)], dtype=np.float32)
                        top = kps2d[:, 1].min()
                        bottom = kps2d[:, 1].max()
                        left = kps2d[:, 0].min()
                        right = kps2d[:, 0].max()
                        top_left = np.array([left, top])
                        bottom_right = np.array([right, bottom])
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max() * self.config['exper']['bbox']['rate']
                        center, legal_indices = undistortion_points(center.reshape(-1, 2), undistortion_param[0],
                                                                    undistortion_param[1],
                                                                    undistortion_param[2],
                                                                    set_bound=True, width=260,
                                                                    height=346)
                        center = center[0]
                    else:
                        top_left = np.array(self.data[seq_id]['annot']['bbox']['event'][str(id)]['tl'], dtype=np.float32)
                        bottom_right = np.array(self.data[seq_id]['annot']['bbox']['event'][str(id)]['br'], dtype=np.float32)
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max()  # * self.config['preprocess']['bbox']['joints_to_bbox_rate'] / 1.5
                    bbox_seq[i, 1:3] = center
                    bbox_seq[i, 3:4] = bbox_size

                bbox_inter_f = interp1d(bbox_seq[:, 0], bbox_seq[:, 1:4], axis=0, kind='quadratic')
                self.bbox_inter[seq_id] = {'event': bbox_inter_f}

    def get_rgb_bbox_matrix_for_fast_sequences(self, rgb_view):
        for seq_id in self.data.keys():
            if self.data[seq_id]['annot']['motion_type'] == 'fast':
                bbox_ids = []
                for key in self.data[seq_id]['annot']['2d_joints'][rgb_view].keys():
                    if self.data[seq_id]['annot']['2d_joints'][rgb_view][str(key)] != []:
                        bbox_ids.append(int(key))
                bbox_ids.sort()
                bbox_seq = np.zeros((len(bbox_ids), 4), dtype=np.float32)
                bbox_seq[:, 0] = np.array(bbox_ids, dtype=np.float32) * 1000000. / 15 + self.data[seq_id]['annot']['delta_time']
                for i, id in enumerate(bbox_ids):
                    if str(id) in self.data[seq_id]['annot']['2d_joints'][rgb_view].keys() and \
                            self.data[seq_id]['annot']['2d_joints'][rgb_view][str(id)] != []:
                        kps2d = np.array(self.data[seq_id]['annot']['2d_joints'][rgb_view][str(id)], dtype=np.float32)
                        top = kps2d[:, 1].min()
                        bottom = kps2d[:, 1].max()
                        left = kps2d[:, 0].min()
                        right = kps2d[:, 0].max()
                        top_left = np.array([left, top])
                        bottom_right = np.array([right, bottom])
                        center = (top_left + bottom_right) / 2.0
                        bbox_size = np.abs(
                            bottom_right - top_left).max() * self.config['exper']['bbox']['rate']
                        center = center
                        bbox_seq[i, 1:3] = center
                        bbox_seq[i, 3:4] = bbox_size
                bbox_inter_f = interp1d(bbox_seq[:, 0], bbox_seq[:, 1:4], axis=0, kind='quadratic')
                self.bbox_inter[seq_id][rgb_view] = bbox_inter_f

    def get_rgb_joint_matrix(self):
        for seq_id in self.data.keys():
            if self.data[seq_id]['annot']['motion_type'] != 'fast':
                joints_ids = [int(id) for id in self.data[seq_id]['annot']['3d_joints'].keys()]
                joints_ids.sort()
                # a tuple (timestamps, 3d joints)
                joints_3ds = np.zeros((len(joints_ids), 21, 3), dtype=np.float32)
                ids_ = np.array(joints_ids, dtype=np.float32)
                ids_timestamp = ids_ * 1000000. / 15 + self.data[seq_id]['annot']['delta_time']
                for i, id in enumerate(joints_ids):
                    joints_3ds[i] = np.array(self.data[seq_id]['annot']['3d_joints'][str(id)], dtype=np.float32).reshape(-1, 3)[indices_change(2, 1)] / 1000.

                # this is 3d joints bbox interpolation function
                bbox_inter_f = interp1d(ids_timestamp, joints_3ds, axis=0, kind='linear')
                self.bbox_inter[seq_id] = {'joints': bbox_inter_f}

    def get_bbox_inter_f(self):
        self.bbox_inter = {}
        self.get_event_bbox_matrix_for_fast_sequences()
        for cam_view in self.data_config['camera_pairs']:
            self.get_rgb_bbox_matrix_for_fast_sequences(cam_view[1])
        self.get_rgb_joint_matrix()

    def get_valid_time(self, x, time):
        if x[0] >= time:
            time_ = x[0]
        elif x[-1] <= time:
            time_ = x[-1]
        else:
            time_ = time
        return time_

    def get_bbox_by_interpolation(self, seq_id, camera_view, timestamp, K=None, R=None, t=None, rate=1.5):
        # get bbox at t0 for fast sequences
        if self.data[seq_id]['annot']['motion_type'] == 'fast':
            timestamp = self.get_valid_time(self.bbox_inter[seq_id][camera_view].x, timestamp)
            bbox = self.bbox_inter[seq_id][camera_view](timestamp)
            bbox = torch.tensor(bbox, dtype=torch.int32)
        else:
            timestamp = self.get_valid_time(self.bbox_inter[seq_id]['joints'].x, timestamp)
            joints = self.bbox_inter[seq_id]['joints'](timestamp)
            joints = torch.tensor(joints, dtype=torch.float32)
            bbox = self.get_bbox_from_joints(K, R, t, joints, rate=rate)
        return bbox

    def get_seq_type(self, seq_id):
        if self.data[seq_id]['annot']['motion_type'] == 'fast':
            return 6
        if self.data[seq_id]['annot']['scene'] == 'normal':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 0
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 1
        elif self.data[seq_id]['annot']['scene'] == 'highlight':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 2
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 3
        elif self.data[seq_id]['annot']['scene'] == 'flash':
            if self.data[seq_id]['annot']['gesture_type'] == 'fixed':
                return 4
            elif self.data[seq_id]['annot']['gesture_type'] == 'random':
                return 5

    def change_camera_view(self, meta_data):
        if self.config['exper']['preprocess']['cam_view'] == 'world':
            return self.get_transform(torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))
        elif self.config['exper']['preprocess']['cam_view'] == 'rgb':
            tf_rgb_w = self.get_transform(meta_data['R_rgb'], meta_data['t_rgb'])
            for mano_key in ['mano', ]:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_rgb_m = tf_rgb_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_rgb_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_rgb_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', ]:
                if joints_key in meta_data.keys():
                    joints_new = (tf_rgb_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_rgb_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for event_cam in ['event', ]:
                tf_e_w = self.get_transform(meta_data['R_'+event_cam], meta_data['t_'+event_cam])
                tf_e_rgb = tf_e_w @ tf_rgb_w.inverse()
                meta_data['R_' + event_cam] = tf_e_rgb[:3, :3]
                meta_data['t_' + event_cam] = tf_e_rgb[:3, 3]
            meta_data['R_rgb'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_rgb'] = torch.zeros(3, dtype=torch.float32)
            return tf_rgb_w.inverse()
        elif self.config['exper']['preprocess']['cam_view'] == 'event':
            tf_event_w = self.get_transform(meta_data['R_event'], meta_data['t_event'])
            for mano_key in ['mano', ]:
                if mano_key in meta_data.keys():
                    R_wm = roma.rotvec_to_rotmat(meta_data[mano_key]['rot_pose'])
                    t_wm = meta_data[mano_key]['trans']
                    tf_wm = self.get_transform(R_wm, t_wm + meta_data[mano_key]['root_joint'])
                    tf_event_m = tf_event_w @ tf_wm
                    meta_data[mano_key]['rot_pose'] = roma.rotmat_to_rotvec(tf_event_m[:3, :3])
                    meta_data[mano_key]['trans'] = tf_event_m[:3, 3] - meta_data[mano_key]['root_joint']
            for joints_key in ['3d_joints', ]:
                if joints_key in meta_data.keys():
                    joints_new = (tf_event_w[:3, :3] @ (meta_data[joints_key].transpose(0, 1))\
                                  + tf_event_w[:3, 3:4]).transpose(0, 1)
                    meta_data[joints_key] = joints_new
            for rgb_cam in ['rgb']:
                tf_rgb_w = self.get_transform(meta_data['R_'+rgb_cam], meta_data['t_'+rgb_cam])
                tf_rgb_e = tf_rgb_w @ tf_event_w.inverse()
                meta_data['R_' + rgb_cam] = tf_rgb_e[:3, :3]
                meta_data['t_' + rgb_cam] = tf_rgb_e[:3, 3]
            meta_data['R_event'] = torch.eye(3, dtype=torch.float32)
            meta_data['t_event'] = torch.zeros(3, dtype=torch.float32)
            return tf_event_w.inverse()
        else:
            raise NotImplementedError('no implemention for change camera view!')

    def get_2d_joints(self, meta_data, ltop, scale):
        target_view = self.config['exper']['preprocess']['cam_view']
        K = meta_data['K_'+target_view]
        R = meta_data['R_'+target_view]
        t = meta_data['t_'+target_view]
        joints = meta_data['3d_joints']
        kps = (K @ (R @ joints.transpose(0, 1) + t.reshape(3, 1))).transpose(0, 1)
        kps = kps[:, :2] / kps[:, 2:]
        kps = (kps - ltop[None]) / scale
        return kps
    
    def get_2d_joints_from_mesh(self, meta_data, ltop, scale):
        manos = meta_data['mano']
        target_view = self.config['exper']['preprocess']['cam_view']
        K = meta_data['K_'+target_view]
        R = meta_data['R_'+target_view]
        t = meta_data['t_'+target_view]
        output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        joints = output.joints
        joints = joints.squeeze(0)
        kps = (K @ (R @ joints.transpose(0, 1) + t.reshape(3, 1))).transpose(0, 1)
        kps = kps[:, :2] / kps[:, 2:]
        kps = (kps - ltop[None]) / scale
        return kps
    
    def __getitem__(self, idx):
        frames_output, meta_data_output = [], []
        seq_id, cam_pair, annot_id = self.get_info_from_sample_id(idx)
        test_fast = self.config['exper']['run_eval_only'] and \
                    self.data[seq_id]['annot']['motion_type'] == 'fast'
        aug_params = self.get_augment_param()

        bbox_valid = True

        if test_fast:
            steps = min(3, self.config['exper']['preprocess']['steps'])
        else:
            steps = self.config['exper']['preprocess']['steps']
        
        rgb_seed = np.random.randint(0, 1000000)
        ev_seed = np.random.randint(0, 1000000)
        for step in range(steps):
            meta_data = self.get_annotations(seq_id, cam_pair, int(annot_id))
            meta_data_prev = self.get_annotations(seq_id, cam_pair, int(annot_id)-1)
            delta_time = meta_data['delta_time']

            if test_fast:
                segments = int(self.config['eval']['fast_fps'] / 15)
            else:
                segments = self.config['exper']['preprocess']['segments']
            if str(annot_id) in self.data[seq_id]['annot']['frames'][cam_pair[1]]:
                rgb_path = osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id, 'images', cam_pair[1],\
                                    'image' + str(annot_id).rjust(4, '0') + '.jpg')
                if not self.config['exper']['run_eval_only']:
                    if torch.rand(1) < self.config['exper']['augment']['rgb_photometry']['motion_blur']:
                        rgb_path = osp.join(self.config['data']['dataset_info']['evrealhands']['data_dir'], seq_id, 'image_blur', cam_pair[1],\
                                    'image' + str(annot_id).rjust(4, '0') + '.jpg')
                rgb = self.load_img(rgb_path)
                rgb_valid = True
            else:
                rgb = np.zeros((920, 1064, 3), dtype=np.float32)
                rgb_valid = False
            meta_data['rgb_valid'] = rgb_valid

            T_ = 1e6 / 15.

            t_target = delta_time + int(annot_id) * T_

            ev_frames = []

            for segment in range(segments):
                t_l = t_target - T_ + segment / segments * T_
                t_r = t_target - T_ + (segment+1) / segments * T_
                
                if self.config['exper']['preprocess']['event_range'] == 'bbox':
                    if test_fast:
                        bbox_ev = self.get_bbox_by_interpolation(seq_id, cam_pair[0], t_r)
                    else:
                        bbox_ev = self.get_bbox_by_interpolation(seq_id, cam_pair[0], t_r, meta_data['K_event'], meta_data['R_event'], meta_data['t_event'], rate=1.5)
                else:
                    bbox_ev=None
                indices_ev = self.get_indices_from_timestamps([t_l, t_r], seq_id, bbox = bbox_ev)
                ev_frame = self.get_event_repre(seq_id, indices_ev)
                # ev_frame = torch.cat([ev_frame.permute(1, 2, 0), torch.zeros((260, 346, 1))], dim=2)
                ev_frames.append(ev_frame)

            if step == 0:
                aff_2d_rgb, tf_cw_rgb = self.get_trans_from_augment(aug_params[0], meta_data, 920, 1064, view='rgb')
                aff_2d_ev, tf_cw_ev = self.get_trans_from_augment(aug_params[0], meta_data, 260, 346,
                                                                          view='event')
            meta_data['R_rgb'] = tf_cw_rgb[:3, :3]
            meta_data['t_rgb'] = tf_cw_rgb[:3, 3]
            meta_data['R_event'] = tf_cw_ev[:3, :3]
            meta_data['t_event'] = tf_cw_ev[:3, 3]

            rgb = cv2.warpAffine(np.array(rgb), aff_2d_rgb, (1064, 920), flags=cv2.INTER_LINEAR)
            rgb, rgb_scene = self.rgb_degrader(torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.)
            rgb = rgb.permute(1, 2, 0)

            for i in range(len(ev_frames)):
                ev_frames[i] = cv2.warpAffine(np.array(ev_frames[i]), aff_2d_ev, (346, 260), flags=cv2.INTER_LINEAR)
                
                ev_frames[i], event_scene = self.event_degrader(torch.tensor(ev_frames[i], dtype=torch.float32).permute(2, 0, 1))
                ev_frames[i] = ev_frames[i].permute(1, 2, 0)

            rate = self.config['exper']['bbox']['rate'] * aug_params[0][3]
            
            if test_fast:
                bbox_rgb = self.get_bbox_by_interpolation(seq_id, cam_pair[1], t_target)
                bbox_evs = []
                for segment in range(segments):
                    t_r = t_target - T_ + (segment + 1) / segments * T_
                    bbox_ev = self.get_bbox_by_interpolation(seq_id, cam_pair[0], t_r)
                    bbox_evs.append(bbox_ev)
            else:
                bbox_rgb = self.get_bbox_by_interpolation(seq_id, cam_pair[1], t_target, meta_data['K_rgb'], meta_data['R_rgb'], meta_data['t_rgb'], rate=rate)
                bbox_evs = []
                for segment in range(segments):
                    t_r = t_target - T_ + (segment + 1) / segments * T_
                    bbox_ev = self.get_bbox_by_interpolation(seq_id, cam_pair[0], t_r, meta_data['K_event'], meta_data['R_event'], meta_data['t_event'], rate=rate)
                    bbox_evs.append(bbox_ev)

            if not self.valid_bbox(bbox_rgb, hw=[920, 1064]):
                bbox_rgb = self.get_default_bbox(hw=[920, 1064], size=self.config['exper']['bbox']['rgb']['size'])
                bbox_valid = False
                print(f'seq_id {seq_id} annot id {annot_id}  rgb bbox invalid')

            for i, bbox_ev in enumerate(bbox_evs):
                if not self.valid_bbox(bbox_ev, hw=[260, 346]):
                    bbox_evs[i] = self.get_default_bbox(hw=[260, 346], size=self.config['exper']['bbox']['event']['size'])
                    bbox_valid = False
                    print(f'seq_id {seq_id} annot id {annot_id}  event bbox invalid')
            
            lt_rgb, sc_rgb, rgb_crop = self.crop(bbox_rgb, np.array(rgb), self.config['exper']['bbox']['rgb']['size'], hw=[920, 1064])
            rgb_crop = torch.tensor(rgb_crop, dtype=torch.float32)
            rgb_crop_un_normal = rgb_crop
            rgb_crop = self.normalize_img(rgb_crop.permute(2, 0, 1)).permute(1, 2, 0)

            ev_frames_crop = []
            lt_evs, sc_evs = [], []
            for i, ev_frame in enumerate(ev_frames):
                lt_ev, sc_ev, ev_frame_crop = self.crop(bbox_evs[i], np.array(ev_frame), self.config['exper']['bbox']['event']['size'],
                                                 hw=[260, 346])
                lt_evs.append(lt_ev)
                sc_evs.append(sc_ev)
                ev_frames_crop.append(torch.tensor(ev_frame_crop, dtype=torch.float32))

            if self.config['exper']['use_2d_joints']:
                target_view = self.config['exper']['preprocess']['cam_view']
                if target_view == 'event':
                    joints_2d = self.get_2d_joints(meta_data, lt_evs[-1], bbox_evs[-1][2])
                    joints_2d_mesh = self.get_2d_joints_from_mesh(meta_data, lt_evs[-1], bbox_evs[-1][2])
                elif target_view == 'rgb':
                    joints_2d = self.get_2d_joints(meta_data, lt_rgb, bbox_rgb[2])
                    joints_2d_mesh = self.get_2d_joints_from_mesh(meta_data, lt_rgb, bbox_rgb[2])
                meta_data['2d_joints'] = joints_2d
                meta_data['2d_joints_mesh'] = joints_2d_mesh


            tf_w_c = self.change_camera_view(meta_data)
            _ = self.change_camera_view(meta_data_prev)

            seq_type = self.get_seq_type(seq_id)
            scene_type = torch.ones(2)
            if seq_type in [4, 5] or event_scene[2] == 1:
                scene_type[0] = 0
            if seq_type in [2,3,6] or rgb_scene[0]==1 or rgb_scene[1] == 1:
                scene_type[1] = 0

            meta_data.update({
                'lt_rgb': lt_rgb,
                'sc_rgb': sc_rgb,
                'lt_evs': lt_evs,
                'sc_evs': sc_evs,
                'scene_weight': scene_type,
                
            })

            frames = {
                'rgb': rgb_crop,
                'ev_frames': ev_frames_crop,
            }
            if self.config['exper']['run_eval_only']:
                frames.update(
                    {
                        'rgb_ori': rgb,
                        'rgb_crop_un_normal': rgb_crop_un_normal,
                        'event_ori': ev_frames,#[torch.tensor(frame, dtype=torch.float32) for frame in ev_frames],
                    }
                )
                meta_data.update({
                    'seq_id': torch.tensor(int(seq_id)),
                    'seq_type': torch.tensor(seq_type),
                    'annot_id': torch.tensor(int(annot_id)),
                    'meta_data_prev': meta_data_prev,
                })
            meta_data.update({
                'tf_w_c': tf_w_c,
            })
            frames_output.append(frames)
            meta_data_output.append(meta_data)
            annot_id = str(int(annot_id) + 1)
        
        supervision_type = 0

        for i in range(len(meta_data_output)):
            meta_data_output[i]['bbox_valid'] = bbox_valid
            meta_data_output[i]['supervision_type'] = supervision_type
        return frames_output, meta_data_output

    def __len__(self):
        return int(self.sample_info['samples_sum'][-1])

    def get_render(self, hw=[920, 1064]):
        self.raster_settings = RasterizationSettings(
            image_size=(hw[0], hw[1]),
            faces_per_pixel=2,
            perspective_correct=True,
            blur_radius=0.,
        )
        self.lights = PointLights(
            location=[[0, 2, 0]],
            diffuse_color=((0.5, 0.5, 0.5),),
            specular_color=((0.5, 0.5, 0.5),)
        )
        self.render = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SoftPhongShader(lights=self.lights)
        )

    def render_hand(self, manos, K, R, t, hw, img_bg=None, joints_2d=None):
        self.get_render(hw)
        output = self.mano_layer(
            global_orient=manos['rot_pose'].reshape(-1, 3),
            hand_pose=manos['hand_pose'].reshape(-1, 45),
            betas=manos['shape'].reshape(-1, 10),
            transl=manos['trans'].reshape(-1, 3)
        )
        # now_vertices = torch.bmm(R.reshape(-1, 3, 3), output.vertices.transpose(2, 1)).transpose(2, 1) + t.reshape(-1, 1, 3)
        # faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(1, 1, 1).type_as(manos['trans'])
        # verts_rgb = torch.ones_like(self.mano_layer.v_template).type_as(manos['trans'])
        # verts_rgb = verts_rgb.expand(1, verts_rgb.shape[0], verts_rgb.shape[1])
        # textures = TexturesVertex(verts_rgb)
        # mesh = Meshes(
        #     verts=now_vertices,
        #     faces=faces,
        #     textures=textures
        # )
        # cameras = cameras_from_opencv_projection(
        #     R=torch.eye(3).repeat(1, 1, 1).type_as(manos['shape']),
        #     tvec=torch.zeros(1, 3).type_as(manos['shape']),
        #     camera_matrix=K.reshape(-1, 3, 3).type_as(manos['shape']),
        #     image_size=torch.tensor([hw[0], hw[1]]).expand(1, 2).type_as(manos['shape'])
        # ).to(manos['trans'].device)

        now_vertices = output.vertices
        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(1, 1, 1).type_as(manos['trans'])
        verts_rgb = torch.ones_like(self.mano_layer.v_template).type_as(manos['trans'])
        verts_rgb = verts_rgb.expand(1, verts_rgb.shape[0], verts_rgb.shape[1])
        textures = TexturesVertex(verts_rgb)
        mesh = Meshes(
            verts=now_vertices,
            faces=faces,
            textures=textures
        )
        cameras = cameras_from_opencv_projection(
            R=R.reshape(-1, 3, 3).type_as(manos['shape']),
            tvec=t.reshape(-1, 3).type_as(manos['shape']),
            camera_matrix=K.reshape(-1, 3, 3).type_as(manos['shape']),
            image_size=torch.tensor([hw[0], hw[1]]).expand(1, 2).type_as(manos['shape'])
        ).to(manos['trans'].device)

        self.render.shader.to(manos['trans'].device)
        res = self.render(
            mesh,
            cameras=cameras
        )
        img = res[..., :3]
        img = img.reshape(-1, hw[0], hw[1], 3)
        # plt.imshow(img[0].detach().cpu().numpy())
        # plt.show()
        if img_bg is not None:
            if img_bg.shape[-1] == 2:
                img_bg = torch.cat([img_bg[...,:1], torch.zeros_like(img_bg[..., :1]), img_bg[...,1:]], dim=-1)
            mask = res[..., 3:4].reshape(-1, hw[0], hw[1], 1) != 0.
            img = torch.clip(img * mask + mask.logical_not() * img_bg[None], 0, 1)
        img_show = img[0].detach().cpu().numpy()
        if joints_2d is not None:
            for joint in joints_2d:
                cv2.circle(img_show, (int(joint[0]), int(joint[1])), 3, (0, 0, 255), -1)
        plt.imshow(img_show)
        plt.show()
        return img
