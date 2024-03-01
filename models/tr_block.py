import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np

import MinkowskiEngine as ME

from common import ConvType, NormType, get_norm, conv, get_nonlinearity_fn
from resnet_block import *

class CodedVTRBlock(nn.Module): # ddp could not contain unused parameter, so donnot inherit from TRBlock
    expansion=1
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

        super(CodedVTRBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        '''
        The Codebook-based Attention: for original feature [1, dim], generate the attnmap [K,h];
        then do dotproduct with codebook [D,M,K,dim], get choice [D,M], use it to aggregate the codebook;
        apply on value [1, dim] to generate final feature
        ------------------
        inplanes/outplanes: the feature dim
        expansion: the width expansion
        qk_type:
            - conv
            - dotproduct(pairwise)
        conv_v: use conv or linear for gen value
        vec_dim: the attn_map feature dim
        H: head num
        D,M - codebook size
        K - neighbor-size

        The Geometry-based Attention
        ------------------
        custom-kernel: use CROSS-like / different dilations of neighbor
        geo-shape: whether apply geo-shape for codebook elements
        temp - the softmax temperature
        ------------------
        '''

        self.expansion = 1
        self.qk_type = 'conv' # ['conv','pairwise']
        self.conv_v = True
        self.top_k_choice = False
        self.temp_ = 1.e0 # the initial temp

        # === some additonal tricks ===
        self.skip_choice = False # only_used in debug mode, notice that this mode contains unused params, so could not support ddp for now
        self.geo_shape = False
        self.sparse_pattern_reg = False

        if self.inplanes != self.planes:
            self.linear_top = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
            self.downsample = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

        if self.conv_v == True:
            self.v = nn.Sequential(
                    MinkoskiConvBNReLU(planes, planes, kernel_size=3),
                    MinkoskiConvBNReLU(planes, planes*self.expansion, kernel_size=1),
                    )
        else:
            self.v = MinkoskiConvBNReLU(planes, planes*self.expansion, kernel_size=1)

        self.codebook = nn.ModuleList([])
        self.D = 3
        self.M = 8
        self.CUSTOM_KERNEL = True
        if self.CUSTOM_KERNEL:  # distinct geometric shape for codebook elements
            kgargs0 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 2,
                # "region_type":ME.RegionType.HYPER_CROSS,
                "region_type":ME.RegionType.HYPER_CUBE,
                "dimension": 3,
                }
            kgargs1 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 1,
                "region_type":ME.RegionType.HYPER_CUBE,
                "dimension": 3,
                }
            kgargs2 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 3,
                "region_type":ME.RegionType.HYPER_CUBE,
                "dimension": 3,
                }
            self.kgargs = [kgargs0, kgargs1, kgargs2] # len should align with M
            kgs = [ME.KernelGenerator(**kg) for kg in self.kgargs]
            for i_ in range(self.D):
                self.codebook.append(
                    nn.Sequential(
                        ME.MinkowskiChannelwiseConvolution(planes*self.expansion, kernel_size=3, dimension=3, kernel_generator=kgs[i_]),
                        )
                    )

            if not self.skip_choice:
                if self.qk_type == 'conv':
                    self.q = nn.ModuleList([])
                    for i_ in range(self.D):
                        self.q.append(
                            nn.Sequential(
                                ME.MinkowskiConvolution(planes,self.M, dimension=3, kernel_generator=kgs[i_]),
                                )
                            )
                elif self.qk_type == 'pairwise':
                    self.q = MinkoskiConvBNReLU(planes, self.M, kernel_size=1)
                    # self.pos_enc = MinkoskiConvBNReLU(3, self.M, kernel_size=1)

        else:
            kgargs0 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 1,
                "region_type":ME.RegionType.HYPER_CUBE,
                "dimension": 3,
                }
            self.kgargs = [kgargs0]*self.D
            for i_ in range(self.D):
                self.codebook.append(
                    nn.Sequential(
                        ME.MinkowskiConvolution(planes,self.M, dimension=3, kernel_generator=kgs[i_]),
                        )
                    )
            if not self.skip_choice:
                if self.qk_type == 'conv':  # since conv already contains the neighbor info, so no pos_enc
                    self.q = nn.Sequential(
                        ME.MinkowskiConvolution(planes, self.M, kernel_size=3,dimension=3),
                        )
                elif self.qk_type == 'pairwise':
                    self.q = MinkoskiConvBNReLU(planes, self.M, kernel_size=1)
                else:
                    raise NotImplementedError

        if self.geo_shape:
            # 3 masks
            # each contains masks at differnt stride
            # mask1 = torch.load('./plot/final/sparse_masks.pth')
            mask0 = np.array([
                    [0,1,3,6,7,13],
                    [1,2,9,14,15,17],
                    [0,5,6,7,8,10],
                    [17,19,20,22,23],
                    ])
            mask1 = np.array([
                    [10,11,12,20,21,22],
                    [1,2,3,4,5,6,10,21,20],
                    [3,4,5,6,7,8,9,10,11],
                    [17,18,19,20,22,23,24],
                    ])
            mask2 = np.array([
                    [0,5,9,13,19,22],
                    [1,3,7,8,11,16,20],
                    [4,6,11,12,18,24,25],
                    [5,6,10,14,19,23],
                    ])

            self.codebook_masks = [mask0, mask1, mask2]

            for _ in range(len(self.codebook)):
                new_kernel = self.codebook[_][0].kernel
                k_, dim_ = new_kernel.shape
                if len(self.codebook_masks[_])>0:
                    assert self.M % len(self.codebook_masks[_]) == 0
                if len(self.codebook_masks[_])>1:
                    dim_per_mask = dim_ // len(self.codebook_masks[_])
                else:
                    dim_per_mask = dim_
                for m_ in range(len(self.codebook_masks[_])):
                    new_kernel[self.codebook_masks[_][m_],dim_per_mask*m_:dim_per_mask*(m_+1)] = 0
                self.codebook[_][0].kernel = nn.Parameter(self.codebook[_][0].kernel)

            # codebook_weight = torch.stack([m[0].kernel for m in self.codebook])

        self.out_bn_relu = nn.Sequential(
                ME.MinkowskiConvolution(planes*self.expansion, planes, kernel_size=1, dimension=3),
                # ME.MinkowskiBatchNorm(planes),
                ME.MinkowskiReLU(),
                )

    def expand_dim(self,x):
        # x shold be like [N, vec_dim]; [N, vec_dim, M]
        # expand em as [N, dim]; [N, dim, M]
        assert x.shape[1] == self.M
        if len(x.shape) == 2:
            N, dim = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.expansion//self.M).reshape(-1,self.planes*self.expansion)
        elif len(x.shape) == 3:
            N, dim, M = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.expansion//self.M, -1).reshape(-1,self.planes*self.expansion,M)

        return x

    def get_sparse_pattern(self, x, choice, type_=1):
        # FORMULA 1: get codebook kernel shapes and directly use the sparse-pattern matching 
        # as the guidance of choice
        if type_ == 1:

            sparse_patterns= []  # [M]
            for m_ in range(self.D):
                kgargs = self.kgargs[m_]
                if 'dimension' in kgargs.keys():
                    del kgargs['dimension']
                neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key,
                                                                    x.coordinate_map_key,
                                                                    **kgargs
                                                                    )
                N = x.C.shape[0]
                # its easy to get how many matched elements of cur-point & kernel
                # but the kernel shape is hard to be flexible, like i need to index the lower-right part
                if self.geo_shape:
                    # only when codebook-prior is given, each point would have different pattern
                    sparse_pattern_ = torch.zeros([N, self.M], device=x.device)
                else:
                    sparse_pattern_ = torch.zeros([N, 1], device=x.device)

                if hasattr(self, "codebook_masks"):
                    # TODO: acquire the stride corresponding 
                    cur_mask = self.codebook_masks[m_]
                else:
                    cur_mask = []

                cur_k = len(neis_d.keys())
                for k_ in range(cur_k):

                    if not k_ in neis_d.keys():
                            continue

                    if len(cur_mask)>0:
                        for i_ in range(len(cur_mask)):
                            if k_ in cur_mask[i_]:  # for masked k
                                continue
                            else:
                                sparse_pattern_[neis_d[k_][0].long(),i_] +=1
                    else:
                        sparse_pattern_[neis_d[k_][0].long(),:] +=1

                if len(cur_mask)>0:
                    for i_ in range(len(cur_mask)):
                        sparse_pattern_[:,i_] = sparse_pattern_[:,i_] / (cur_k - len(cur_mask[i_]))
                        if cur_k == len(cur_mask[i_]): # assert zero division, empty kernel
                            import ipdb; ipdb.set_trace()
                else:
                    sparse_pattern_ = sparse_pattern_ / cur_k
                sparse_patterns.append(sparse_pattern_)
            sparse_patterns = torch.stack(sparse_patterns, dim=-1) # [N,D,M]
            # Reg Type1:  encourage the kernel to lean to map with more matching neighbors
            temp_ = 0.2
            eps = 1.e-3
            sparse_patterns = F.softmax((F.softmax((sparse_patterns+eps)/temp_, dim=1)+eps)/temp_, dim=-1)  # [N. vec-dim. M] 
            self.register_buffer("sparse_patterns",sparse_patterns)

            return choice*self.sparse_patterns
        else:
            # formula 2: MultiScale Estimation of how sparse a point is 
            # apply softmax in the normalized N points dimension
            # calc the relative sparsity distance to many centers as regs
            raise NotImplementedError

    # def schedule_update(self, iter_=0):
    #     '''
    #     some schedulable params
    #     '''
    #     # ======= the temp annealing for choice =============
    #     self.temp = (self.temp_)**(1-iter_) # start from the temp, end with 0
    #
    #     if self.skip_choice == True and iter_> 0.1:
    #         self.skip_choice = False
    #         print('SkipChoice Warmup Done, Start training choice qk')
    #
    #     if self.skip_choice == False and not hasattr(self, "q"):
    #         self.q = nn.Sequential(
    #             ME.MinkowskiConvolution(self.planes, self.M, kernel_size=3,dimension=3),
    #             ME.MinkowskiBatchNorm(self.M),
    #                 )
    #         self.q.to(self.codebook[0][0].kernel.device)
    #
    #     pass
    #
    #     # ========= Temperature Annealing ==============

        # if not hasattr(self, 'temp0'):
            # self.temp0 = self.temp

        # self.temp = self.temp0*(0.01)**(iter_)

    def forward(self, x, iter_=None, aux=None):
        '''
        For each dilation(D=3), different d have different kernel shape and different Ks, e.g., cube-shape kernel has k=27, cross-shaped has k=7
        1st do qk projection: [N, dim, K]  ()
                - conv: directly use conv neighbor aggregation(extra params), output: [N, H]
                - pairwise: use linear mapping, then gather neighbor & dotproduct. output: [N, H, K] -> [N, H]
        2nd: q_ dot product with Codebook(M set of conv weights): [N, H, M] -> [N, dim, M], the apply softmax to get choice of [D, M]
        3rd: use choice: [D, M] to aggregate M codebook elements(channel-wise convs) for each point, then apply the coedbook(through channel-wise conv on value)
        '''
        self.register_buffer('coord_map', x.C)
        # self.schedule_update(iter_)

        # align the channel for the decoder that concat the input
        if self.planes != self.inplanes:
            res = self.downsample(x)
            x = self.linear_top(x)
        else:
            res = x

        # generate the value
        v_ = self.v(x)

        # generate the qk
        if self.skip_choice:
            pass
        else:  # no skip choice
            if self.qk_type == 'conv':
                if not self.CUSTOM_KERNEL:
                    q_ = self.q(x)
                    q_f = self.expand_dim(q_.F)
                    q_= ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]
                    N, dim = q_.F.shape
                    qs = [q_]*self.D
                else:
                    qs = []
                    for _ in range(self.D):
                        q_ = self.q[_](x)
                        q_f =self.expand_dim(q_.F)
                        qs.append(
                            ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]
                                )
                        N, dim = q_f.shape

                # get dot-product of codebook-weight & q_
                choice = []
                out = []
                for _ in range(self.D):
                    self.codebook[_][0].kernel.requires_grad = False   # detach the grad from choice to codebook elements
                    choice_ = self.codebook[_](qs[_])
                    choice.append(choice_.F.reshape(
                        [choice_.shape[0], self.M, self.planes*self.expansion // self.M] #[N,M,c]
                            ).sum(-1)
                        )
                choice = torch.stack(choice, dim=-1)
                eps = 1.e-3

                if self.D > 1: # if M==1, skip softmax since there is only 1 value
                    choice = F.softmax((choice)/1, dim=-1) # [N, vec_dim, M]
                else:
                    pass

                # attn_map = torch.stack([self.codebook[_][0].kernel for _ in range(self.D) ], dim=0) # [M. K], in some case(CUSTOM_KERNEL)
                attn_map = torch.cat([self.codebook[_][0].kernel for _ in range(self.D)],dim=0) # [M. K]
                self.register_buffer('attn_map', attn_map)
                self.register_buffer('choice_map', choice)

            elif self.qk_type == 'pairwise':

                q_ = self.q(x)
                q_f = q_.F
                N, _ = q_.F.shape

                choices = []
                for i_m, kg in enumerate(self.kgargs):  # iter over M
                    if 'dimension' in kg.keys():
                        del kg['dimension']
                    neis_d = q_.coordinate_manager.get_kernel_map(q_.coordinate_map_key,
                                                                    q_.coordinate_map_key,
                                                                    **kg
                                                                        )
                    choice = []
                    for k_ in range(len(neis_d.keys())):
                        if not k_ in neis_d.keys():
                            continue
                        neis_ = torch.gather(q_.F, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.M).long())
                        neis = torch.zeros(N,self.M, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                        neis.scatter_(dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.M).long(), src=neis_)

                        sparse_mask_cur_k = (neis.abs().sum(-1) > 0).float()
                        neis = neis*(q_.F*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.M))
                        neis = neis*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.M)

                        out_cur_k = self.expand_dim(neis)*self.codebook[i_m][0].kernel[k_].unsqueeze(0)
                        out_cur_k = out_cur_k.sum(1)  # [N]
                        choice.append(out_cur_k)

                    choice = torch.stack(choice, dim=-1)  # [N,K]
                    choice = F.softmax(choice/self.temp, dim=-1).sum(-1)
                    choices.append(choice) # [N]

                choices = torch.stack(choices, dim=-1)
                choices = F.softmax(choices/self.temp, dim=-1)   # [N,M]
                choice = choices.unsqueeze(1).expand(-1, self.M, -1) # [N, dim, M]
                self.register_buffer('choice_map', choices)

            if self.sparse_pattern_reg:
                choice = self.get_sparse_pattern(x, choice)

        if self.skip_choice:
            N, dim = v_.shape
            out = []
            for _ in range(self.D):
                self.codebook[_][0].kernel.requires_grad = True
                out_ = self.codebook[_](v_)
                out.append(out_.F)
            out = torch.stack(out, dim=-1)
            out = out.sum(-1)

        elif self.top_k_choice:
            assert self.M == 1 # same point use the same choice 
            out = torch.zeros([N,dim,self.top_k_choice], device=x.device)
            choice_topk = torch.topk(choice, self.top_k_choice, dim=-1)[0] # shape [N,dim]
            choice_topk_idx = torch.topk(choice, self.top_k_choice, dim=-1)[1][:,0,:]  # shape [N]
            for _ in range(self.D):
                self.codebook[_][0].kernel.requires_grad = True
                # DEV: split points for different choice
                # however, if choice has the channle freedom
                # could not handle
                cur_out_ = self.codebook[_](v_) # the conv
                for top_ in range(self.top_k_choice):
                    choice_idx = torch.where(choice_topk_idx[:,top_] == _)[0]
                    # cur_v_ = v_.features_at_coordinates(v_.C[choice_idx,:].float())
                    if len(choice_idx) > 1:
                        # cur_v_ = ME.SparseTensor(
                                # features=v_.F[choice_idx,:],
                                # coordinates=v_.C[choice_idx,:],
                                # coordinate_map_key=v_.coordinate_map_key,
                                # coordinate_manager=v_.coordinate_manager
                                # )
                        out[:,:,top_].scatter_(
                                src=cur_out_.F[choice_idx,:]*choice_topk[choice_idx,:,top_],
                                index=choice_idx.unsqueeze(-1).repeat(1,dim),
                                dim=0)
                    else:
                        pass
            out = out.sum(-1)
        else:
            # normal-case: apply the attn_weight aggregation with the channelwiseConvolution
            out = torch.zeros([N, self.planes*self.expansion], device=v_.device)
            for _ in range(self.D):
                self.codebook[_][0].kernel.requires_grad = True
                out_ = self.codebook[_](v_)
                out += out_.F*self.expand_dim(choice[:,:,_])
            out = out.reshape([N, self.planes*self.expansion])

        out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        out = self.out_bn_relu(out)
        out = out + res

        return out

