# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch.nn as nn
import torch
from torch.autograd import Function

import MinkowskiEngine as ME

from common import ConvType, NormType, get_norm, conv, get_nonlinearity_fn

def separate_batch(coord: torch.Tensor):
    """
        Input:
            coord: (N_voxel, 4) coordinate tensor, coord=b,x,y,z
        Return:
            tensor: (B, N(max n-voxel cur batch), 3), batch index separated
            mask: (B, N), 1->valid, 0->invalid
    """

    # Features donot have batch-ids
    N_voxel = coord.shape[0]
    B = (coord[:,0].max().item() + 1)

    batch_ids = coord[:,0]

    # get the splits of different i_batchA
    splits_at = torch.stack([torch.where(batch_ids == i)[0][-1] for i in torch.unique(batch_ids)]).int() # iter at i_batch_level
    # the returned indices of torch.where is from [0 ~ N-1], but when we use the x[start:end] style indexing, should cover [1:N]
    # example: x[0:1] & x[:1] are the same, contain 1 element, but x[:0] is []
    # example: x[:N] would not raise error but x[N] would

    splits_at = splits_at+1
    splits_at_leftshift_one = splits_at.roll(shifts=1)   # left shift the splits_at
    splits_at_leftshift_one[0] = 0

    len_per_batch = splits_at - splits_at_leftshift_one
    # len_per_batch[0] = len_per_batch[0]+1 # DBEUG: dirty fix since 0~1566 has 1567 values
    N = len_per_batch.max().int()

    assert len_per_batch.sum() == N_voxel

    mask = torch.zeros([B*N], device=coord.device).int()
    new_coord = torch.zeros([B*N, 3], device=coord.device).int() # (B, N, xyz)

    '''
    new_coord: [B,N,3]
    coord-part : [n_voxel, 3]
    idx: [b_voxel, 3]
    '''
    idx_ = torch.cat([torch.arange(len_, device=coord.device)+i*N for i, len_ in enumerate(len_per_batch)])
    idx = idx_.reshape(-1,1).repeat(1,3)
    new_coord.scatter_(dim=0, index=idx, src=coord[:,1:])
    mask.scatter_(dim=0, index=idx_, src=torch.ones_like(idx_, device=idx.device).int())
    mask = mask.reshape([B,N])
    new_coord = new_coord.reshape([B,N,3])

    return new_coord, mask, idx_

def voxel2points_(x_c, x_f_):
    '''
    pack the ME Sparse Tensor feature(batch-dim information within first col of coord)
    [N_voxel_all_batches, dims] -> [bs, max_n_voxel_per_batch, dim]

    idx are used to denote the mask
    '''

    x_c, mask, idx = separate_batch(x_c)
    B = x_c.shape[0]
    N = x_c.shape[1]
    dim = x_f_.shape[1]
    idx_ = idx.reshape(-1,1).repeat(1,dim)
    x_f = torch.zeros(B*N, dim).cuda()
    x_f.scatter_(dim=0, index=idx_, src=x_f_)
    x_f = x_f.reshape([B,N,dim])

    return x_c, x_f, idx

def points2voxel(x, idx):
    '''
    revert the points into voxel's feature
    returns the new feat
    '''
    # the origi_x provides the cooed_map
    B, N, dim = list(x.shape)
    new_x = torch.gather(x.reshape(B*N, dim), dim=0, index=idx.reshape(-1,1).repeat(1,dim))
    return new_x

def gen_pos_enc(x_c, x_f, neighbor, mask, idx_, delta, rel_xyz_only=False, register_map=False):
    k = neighbor.shape[1]
    try:
        relative_xyz = neighbor - x_c[:,None,:].repeat(1,k,1) # (nvoxel, k, bxyz), we later pad it to [B, xyz, nvoxel_batch, k]
    except:
        import ipdb; ipdb.set_trace()
    relative_xyz[:,0,0] = x_c[:,0] # get back the correct batch index, because we messed batch index in the subtraction above
    relative_xyz = pad_zero(relative_xyz, mask) # [B, xyz, nvoxel_batch, k]

    pose_tensor = delta(relative_xyz.float()) # (B, feat_dim, nvoxel_batch, k)
    pose_tensor = make_position_tensor(pose_tensor, mask, idx_, x_c.shape[0]) # (nvoxel, k, feat_dim)S
    # if self.SUBSAMPLE_NEIGHBOR:
        # pose_tensor = pose_tensor[:,self.perms,:]
    # if register_map:
        # self.register_buffer('pos_map', pose_tensor.detach().cpu().data)
    if rel_xyz_only:
        pose_tensor = make_position_tensor(relative_xyz.float(), mask, idx_, x_c.shape[0]) # (nvoxel, k, feat_dim)
    return pose_tensor


def get_sparse_neighbor(k, x, kernel_size=3, stride=1, additional_xf=None):

    if additional_xf is not None:
        x_f = additional_xf
    else:
        x_f = x.F
    N, dim = x_f.shape
    neis = torch.zeros(N,k,dim, device=x_f.device)
    rel_xyz = torch.zeros(N,k,3, device=x.C.device)
    neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key, x.coordinate_map_key,kernel_size=kernel_size, stride=stride)
    for k_ in range(k):
        if not k_ in neis_d.keys():
            continue
        # tmp_neis = torch.zeros(N,dim, device=x.F.device)
        neis_ = torch.gather(x_f, dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,dim).long())
        neis[:,k_,:] = torch.scatter(neis[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,dim).long(), src=neis_)
        rel_xyz_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long())
        rel_xyz[:,k_,:] = torch.scatter(rel_xyz[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long(), src=rel_xyz_)

    # if additional_xf is not None:
        # neis = neis.permute(0,2,1).squeeze(2)

    # N, dim
    return neis, rel_xyz

class apply_choice(Function):
    @staticmethod
    def forward(ctx, out, choice):
        ctx.save_for_backward(out, choice)
        return out*choice

    @staticmethod
    def backward(ctx, g):
        out, choice = ctx.saved_tensors
        g_out = g*torch.ones_like(out, device=out.device) # skip grad of choice on out
        g_choice = (g*out).sum(1).unsqueeze(2)
        return g_out, g_choice

def MinkoskiConvBNReLU(inplanes, planes, kernel_size=1):
    return nn.Sequential(
            ME.MinkowskiConvolution(inplanes, planes, kernel_size=kernel_size, dimension=3),
            # ME.MinkowskiBatchNorm(planes),
            ME.MinkowskiReLU(),
            )

class SingleConv(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(SingleConv, self).__init__()

    self.inplanes = inplanes
    self.planes = planes

    # self.conv = ParameterizedConv(
    self.conv = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    if self.downsample is not None:
        self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None, aux=None):
    residual = x
    
    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

class SingleChannelConv(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(SingleChannelConv, self).__init__()

    self.inplanes = inplanes
    self.planes = planes

    # self.conv = ParameterizedConv(
    # self.conv = conv(
        # inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    if inplanes != planes:
        self.conv = nn.Sequential(
                MinkoskiConvBNReLU(inplanes, planes, kernel_size=1),
                ME.MinkowskiConvolution(planes, planes, kernel_size=1, dimension=3),
                ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                )
    else:
        self.conv = nn.Sequential(
                MinkoskiConvBNReLU(planes, planes, kernel_size=1),
                ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                )


    self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    if self.downsample is not None:
        self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):
    residual = x
    
    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

class ConvBase(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(ConvBase, self).__init__()

    self.conv = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    assert self.downsample is None   # we should not use downsample here
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):
    residual = x

    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        bias=False,
        conv_type=conv_type,
        D=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None, aux=None):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out


class BasicBlock(BasicBlockBase):
  NORM_TYPE = NormType.BATCH_NORM


class BottleneckBase(nn.Module):
  expansion = 4
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(BottleneckBase, self).__init__()
    self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv2 = conv(
        planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
    self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)

    self.downsample = downsample
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv2(out)
    out = self.norm2(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv3(out)
    out = self.norm3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out


class Bottleneck(BottleneckBase):
  NORM_TYPE = NormType.BATCH_NORM
