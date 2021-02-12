import numpy as np

import torch
from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
import numpy as np
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe

def RT2QT(poses_in, mean_t, std_t):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :return: processed poses (translation + quaternion) N x 7
  """
  poses_out = np.zeros((len(poses_in), 7))
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3]
    # q = txq.mat2quat(np.dot(align_R, R))
    q = txq.mat2quat(R)
    q = q/(np.linalg.norm(q) + 1e-12) # normalize
    q *= np.sign(q[0])  # constrain to hemisphere
    # q = qlog(q)
    poses_out[i, 3:] = q
    # t = poses_out[i, :3] - align_t
    # poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()
  # normalize translation
  poses_out[:, :3] -= mean_t
  poses_out[:, :3] /= std_t
  
  return poses_out

def vdot(v1, v2):
  """
  Dot product along the dim=1
  :param v1: N x T x d
  :param v2: N x T x d
  :return: N x T x 1
  """
  out = torch.mul(v1,v2)
  # out = torch.sum(out, 1)
  out = torch.sum(out, dim=-1, keepdim=True)
  return out

def qexp_t(q):
  """
  Applies exponential map to log quaternion
  :param q: N x 3
  :return: N x 4
  """
  n = torch.norm(q, p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q * torch.sin(n)
  q = q / n
  q = torch.cat((torch.cos(n), q), dim=1)
  return q

def qlog_t(q):
  """
  Applies the log map to a quaternion
  :param q: N x 4
  :return: N x 3
  """
  n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
  q = q / n
  return q


def calc_vo_relative_logq(p0, p1):
  """
  Calculates VO (in the world frame) from 2 poses (log q)
  :param p0: N x 6
  :param p1: N x 6
  :return:
  """
  q0 = qexp_t(p0[:, 3:])
  q1 = qexp_t(p1[:, 3:])
  vos = calc_vo_relative(torch.cat((p0[:, :3], q0), dim=1), torch.cat((p1[:, :3], q1), dim=1))
  vos_q = qlog_t(vos[:, 3:])
  return torch.cat((vos[:, :3], vos_q), dim=1)


def calc_vo_relative(p0, p1):
  """
  calculates VO (in the world frame) from 2 poses
  :param p0: N x 7
  :param p1: N x 7
  """
  vos_t = p1[:, :3] - p0[:, :3]
  vos_q = qmult(qinv(p0[:, 3:]), p1[:, 3:])
  return torch.cat((vos_t, vos_q), dim=1)

def normalize(x, p=2, dim=0):
  """
  Divides a tensor along a certain dim by the Lp norm
  :param x: 
  :param p: Lp norm
  :param dim: Dimension to normalize along
  :return: 
  """
  xn = x.norm(p=p, dim=dim)
  x = x / xn.unsqueeze(dim=dim)
  return x


def qmult(q1, q2):
  """
  Multiply 2 quaternions
  :param q1: Tensor N x 4
  :param q2: Tensor N x 4
  :return: quaternion product, Tensor N x 4
  """
  q1s, q1v = q1[:, :1], q1[:, 1:]
  q2s, q2v = q2[:, :1], q2[:, 1:]
  qs = q1s*q2s - vdot(q1v, q2v)
  qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
  q  = torch.cat((qs, qv), dim=1)

  # normalize
  q = normalize(q, dim=1)
  return q


def qinv(q):

  """
  Inverts quaternions
  :param q: N x 4
  :return: q*: N x 4 
  Note: q is a unit quaternion !!
  """
  # q_inv = np.zeros(q.shape)
  # q_inv[1 ,:] = q[1]
  # q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
  q_inv = torch.from_numpy(np.concatenate((q[:, :1], -q[:, 1:]), axis=1))

  # s = q.size()
  # q_inv = torch.cat((q[:,..., :1], -q[:,..., 1:]), dim=-1)
  return q_inv
