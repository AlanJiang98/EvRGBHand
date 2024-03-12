"""
Functions for compuing Procrustes alignment and reconstruction error

Parts of the code are adapted from https://github.com/akanazawa/hmr

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T
    return S1_hat
# def compute_similarity_transform(mtx2, mtx1):
#     """ Align the predicted entity in some optimality sense with the ground truth. """
#     # center
#     t1 = mtx1.mean(0)
#     t2 = mtx2.mean(0)
#     mtx1_t = mtx1 - t1
#     mtx2_t = mtx2 - t2

#     # scale
#     s1 = np.linalg.norm(mtx1_t) + 1e-8
#     mtx1_t /= s1
#     s2 = np.linalg.norm(mtx2_t) + 1e-8
#     mtx2_t /= s2

#     # orth alignment
#     R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

#     # apply trafos to the second matrix
#     mtx2_t = np.dot(mtx2_t, R.T) * s
#     mtx2_t = mtx2_t * s1 + t1

#     return mtx2_t


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    is_tensor = False
    if type(S1) == torch.Tensor:
        S1 = S1.cpu().numpy()
        S2 = S2.cpu().numpy()
        is_tensor = True
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    if is_tensor:
        S1_hat = torch.tensor(S1_hat)
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


def reconstruction_error_v2(S1, S2, J24_TO_J14, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    S1_hat = S1_hat[:,J24_TO_J14,:]
    S2 = S2[:,J24_TO_J14,:]
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

def get_alignMesh(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, S1_hat, S2
