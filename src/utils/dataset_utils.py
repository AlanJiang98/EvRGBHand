"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""


import os
import os.path as op
import numpy as np
import base64
import cv2
import yaml
from collections import OrderedDict
import json
from dv import AedatFile
from dv import LegacyAedatFile
import torch


def img_from_base64(imagestring):
    try:
        jpgbytestring = base64.b64decode(imagestring)
        nparr = np.frombuffer(jpgbytestring, np.uint8)
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return r
    except:
        return None


def load_labelmap(labelmap_file):
    label_dict = None
    if labelmap_file is not None and op.isfile(labelmap_file):
        label_dict = OrderedDict()
        with open(labelmap_file, 'r') as fp:
            for line in fp:
                label = line.strip().split('\t')[0]
                if label in label_dict:
                    raise ValueError("Duplicate label " + label + " in labelmap.")
                else:
                    label_dict[label] = len(label_dict)
    return label_dict


def load_shuffle_file(shuf_file):
    shuf_list = None
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            shuf_list = []
            for i in fp:
                shuf_list.append(int(i.strip()))
    return shuf_list


def load_box_shuffle_file(shuf_file):
    if shuf_file is not None:
        with open(shuf_file, 'r') as fp:
            img_shuf_list = []
            box_shuf_list = []
            for i in fp:
                idx = [int(_) for _ in i.strip().split('\t')]
                img_shuf_list.append(idx[0])
                box_shuf_list.append(idx[1])
        return [img_shuf_list, box_shuf_list]
    return None


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def mkdir(path):
    if os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    else:
        raise FileNotFoundError('{} is not a legal direction.'.format(path))


def json_read(file_path):
    with open(os.path.abspath(file_path)) as f:
        data = json.load(f)
        return data
    raise ValueError("Unable to read json file: {}".format(file_path))


def json_write(file_path, data):
    directory = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        with open(os.path.abspath(file_path), 'w') as f:
            json.dump(data, f)
    except Exception:
        raise ValueError("Unable to write json file: {}".format(file_path))

def extract_data_from_aedat2(path: str):
    '''
    extract events from aedat2 data
    :param path:
    :return:
    '''
    with LegacyAedatFile(path) as f:
        events = []
        for event in f:
            events.append(np.array([[event.x, event.y, event.polarity, event.timestamp]], dtype=np.float32))
        if not events:
            return None
        else:
            events = np.vstack(events)
            return events
    return FileNotFoundError('Path {} is unavailable'.format(path))


def extract_data_from_aedat4(path: str, is_event: bool = True, is_aps: bool=False, is_trigger: bool=False):
    '''
    :param path: path to aedat4 file
    :return: events numpy array, aps numpy array
        event:
        # Access information of all events by type
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        is_aps: list of frames
        is_trigger: list of triggers
        # EXTERNAL_INPUT_RISING_EDGE->2, EXTERNAL_INPUT1_RISING_EDGE->6, EXTERNAL_INPUT2_RISING_EDGE->9
        # EXTERNAL_INPUT1_PULSE->8, TIMESTAMP_RESET->1, TIMESTAMP_WRAP->0, EXTERNAL_INPUT1_FALLING_EDGE->7
    '''
    with AedatFile(path) as f:
        events, frames, triggers = None, [], [[], [], [], [], [], [], []]
        id2index = {2: 0, 6: 1, 9: 2, 8: 3, 1: 4, 0: 5, 7: 6}
        if is_event:
            events = np.hstack([packet for packet in f['events'].numpy()])
        if is_aps:
            for frame in f['frames']:
                frames.append(frame)
        if is_trigger:
            for i in f['triggers']:
                if i.type in id2index.keys():
                    triggers[id2index[i.type]].append(i)
                else:
                    print("{} at {} us is the new trigger type in this aedat4 file".format(i.type, i.timestamp))
        return events, frames, triggers
    return FileNotFoundError('Path {} is unavailable'.format(path))

def undistortion_points(xy, K_old, dist, K_new=None, set_bound=False, width=346, height=240):
    '''
    :param xy: N*2 array of event coordinates
    :param K_old: camera intrinsics
    :param dist: distortion coefficients
        such as
        mtx = np.array(
            [[252.91294004, 0, 129.63181808],
            [0, 253.08270535, 89.72598511],
            [0, 0, 1.]])
        dist = np.array(
            [-3.30783118e+01,  3.40196626e+02, -3.19491618e-04, -6.28058571e-04,
            1.67319020e+02, -3.27436981e+01,  3.29048638e+02,  2.85123812e+02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00])
    :param K_new: new K for camera intrinsics
    :param set_bound: if true, set the undistorted points bounds
    :return: undistorted points
    '''
    # this function only outputs the normalized point coordinated, so we need apply a projection matrix K
    assert (xy.shape[1] == 2)
    xy = xy.astype(np.float32)
    if K_new is None:
        K_new = K_old
    und = cv2.undistortPoints(src=xy, cameraMatrix=K_old, distCoeffs=dist, P=K_new)
    und = und.reshape(-1, 2)
    und = und[:, :2]
    legal_indices = (und[:, 0] >= 0) * (und[:, 0] <= width-1) * (und[:, 1] >= 0) * (und[:, 1] <= width-1)
    if set_bound:
        und[:, 0] = np.clip(und[:, 0], 0, width-1)
        und[:, 1] = np.clip(und[:, 1], 0, height-1)
    return und, legal_indices


def remove_unfeasible_events(events, height, width):
    x_mask = (events[:, 0] >=0) * (events[:, 0] <= width-1)
    y_mask = (events[:, 1] >=0) * (events[:, 1] <= height-1)
    mask = x_mask * y_mask
    return mask


def get_intepolate_weight(events, weight, height, width, is_LNES=False):
    top_y = torch.floor(events[:, 1:2])
    bot_y = torch.floor(events[:, 1:2] + 1)
    left_x = torch.floor(events[:, 0:1])
    right_x = torch.floor(events[:, 0:1] + 1)

    top_left = torch.cat([left_x, top_y], dim=1)
    top_right = torch.cat([right_x, top_y], dim=1)
    bottom_left = torch.cat([left_x, bot_y], dim=1)
    bottom_right = torch.cat([right_x, bot_y], dim=1)

    idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
    events_tmp = torch.cat([events for i in range(4)], dim=0)
    zeros = torch.zeros(idx.shape).type_as(idx)
    weights_bi = torch.max(zeros, 1 - torch.abs(events_tmp[:, :2] - idx))
    mask = remove_unfeasible_events(idx, height, width)
    weight_ori = torch.cat([weight for i in range(4)], dim=0)
    events_tmp[:, :2] = idx
    if is_LNES:
        weights_bi_tmp = torch.prod(weights_bi, dim=-1)
        mask *= (weights_bi_tmp != 0)
        weights_lnes = events_tmp[:, 3][mask]
        events_tmp = events_tmp[mask]
        weights_final, indices = torch.sort(weights_lnes, dim=0, descending=False)
        events_final = events_tmp[indices]
        return events_final, weights_final
    else:
        weights_final = torch.prod(weights_bi, dim=-1) * mask * weight_ori
        return events_tmp[mask], weights_final[mask]


def event_to_LNES(event_tmp, height=260, width=346, count=False, interpolate=False):
    ts = event_tmp[:, 3]
    if count:
        img = torch.zeros((3, height, width), dtype=torch.float32)
    else:
        img = torch.zeros((2, height, width), dtype=torch.float32)
    if interpolate:
        events, weights = get_intepolate_weight(event_tmp, ts, height, width, is_LNES=True)
    else:
        events, weights = event_tmp, ts
    if events.dtype is not torch.long:
        xyp = events[:, :3].clone().long()
    # img[xyp[:, 2], xyp[:, 1], xyp[:, 0]] = weights
    # img_new = img.clone()
    img_new_arg = img  # .clone()
    # t_s = time.time()
    # img[xyp[:, 2], xyp[:, 1], xyp[:, 0]] = weights
    # t_1 = time.time() - t_s
    # print('time 1', t_1)
    # t_s = time.time()
    # for i in range(xyp.shape[0]):
    #     img_new[xyp[i, 2], xyp[i, 1], xyp[i, 0]] = weights[i]
    # t_2 = time.time() - t_s
    # print('time 2', t_2)

    # t_s = time.time()
    xyp_ = xyp.numpy()
    weights_ = weights.numpy()
    eventforsort = np.zeros(xyp_.shape[0],
                            dtype=[('p', xyp_.dtype), ('y', xyp_.dtype), ('x', xyp_.dtype), ('w', weights_.dtype)])
    eventforsort[:]['p'] = xyp_[:, 2]
    eventforsort[:]['x'] = xyp_[:, 0]
    eventforsort[:]['y'] = xyp_[:, 1]
    eventforsort[:]['w'] = weights_
    indices = np.argsort(eventforsort, order=('p', 'y', 'x', 'w'))
    xyp_ = xyp_[indices]  #
    weights_ = weights_[indices]
    y_bias = xyp_[1:] - xyp_[:-1]
    breakpoints = np.where(np.sum(np.abs(y_bias), axis=1) != 0)
    # breakpoints = np.concatenate([breakpoints[0], np.array([xyp_.shape[0] - 1])])
    # events[:,1]
    xyp = torch.tensor(xyp_[breakpoints], dtype=torch.long)
    weights = torch.tensor(weights_[breakpoints])
    # xyp = xyp[breakpoints]
    # weights = weights[breakpoints]
    img_new_arg[xyp[:, 2], xyp[:, 1], xyp[:, 0]] = weights
    img_new_arg[xyp_[-1, 2].item(), xyp_[-1, 1].item(), xyp_[-1, 0].item()] = weights_[-1].item()
    # t_3 = time.time() - t_s
    # print('time 3', t_3)
    # print('yes?', (img == img_new).all())
    # print('yes??', (img_new_arg == img_new).all())
    img = img_new_arg
    if count:
        #todo repre check
        weight_polarity = event_tmp[:, 2] * 2 - 1
        if interpolate:
            xys, weights = get_intepolate_weight(event_tmp[:, :2], weight_polarity.float(), height, width)
        else:
            xys, weights = event_tmp[:, :2].clone().long(), weight_polarity
        img[2].index_put_((xys[:, 1], xys[:, 0]), weights, accumulate=True)
        img[2] = torch.clip(img[2] / 2, -1, 1)
        return img[[1, 0, 2], :, :]
    else:
        return img[[1, 0], :, :]


def event_count_to_frame(xy, weight, height=260, width=346, interpolate=False):
    img = torch.zeros((height, width), dtype=torch.float32).to(xy.device)
    if interpolate:
        xys, weights = get_intepolate_weight(xy, weight, height, width)
    else:
        xys, weights = xy, weight
    if xys.dtype is not torch.long:
        xys = xys.clone().long()
    img.index_put_((xys[:, 1], xys[:, 0]), weights, accumulate=True)
    return img


def event_to_channels(event_tmp, height=260, width=346, is_neg=False, interpolate=False):
    mask_pos = event_tmp[:, 2] == 1
    mask_neg = event_tmp[:, 2] == 0
    pos_img = event_count_to_frame(event_tmp[:, :2], (1*mask_pos).float(), height, width, interpolate)
    neg_img = event_count_to_frame(event_tmp[:, :2], (1*mask_neg).float(), height, width, interpolate)
    if is_neg:
        neg_img *= -1
    return torch.stack([pos_img, neg_img])


def event_representations(events, repre='LNES', hw=(260, 346)):
    if repre == 'LNES':
        ev_frame = event_to_LNES(events, height=hw[0], width=hw[1])
    if repre == 'eci':
        ev_frame = event_to_channels(events, height=hw[0], width=hw[1])
    if repre == 'LNES-Count':
        ev_frame = event_to_LNES(events, height=hw[0], width=hw[1], count=True)
        pass
    return ev_frame
