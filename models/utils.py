import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from tr_block import CodedVTRBlock

class enconder(nn.Module):
    def __init__(self):
        super(enconder, self).__init__()
        self.conv0=ME.MinkowskiConvolution(
            3,
            64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0=ME.MinkowskiConvolution(
            64,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.conv1=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.conv2=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.conv3= ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.bn1=ME.MinkowskiBatchNorm(128)
        # self.bn2=ME.MinkowskiBatchNorm(128)
        # self.bn3=ME.MinkowskiBatchNorm(128)

        self.TCM=TCM(128)
        self.GCM=GCM(128)
        self.TCM1 = TCM(128)
        self.GCM1 = GCM(128)
        # for p in self.parameters():
        #     p.requires_grad=False
    def forward(self, x):
        x=self.relu((self.down0(self.conv0(x))))
        x=self.GCM(self.TCM(self.relu((self.down1(self.conv1(x))))))
        x=self.conv3(self.GCM1(self.TCM1(self.relu(self.down2(self.conv2(x))))))


        return x


class deconder(nn.Module):
    def __init__(self):
        super(deconder, self).__init__()
        # self.first_layer = first_layer
        # if first_layer:CONDA
        #     self.fa1 = FeatureEmbeddingModule(nsample, radius, in_channel, mlp, shortcut=False)
        # else:
        #     self.fa1 = FeatureEmbeddingModule(nsample, radius, in_channel, mlp, shortcut=True)
        # self.fa2 = FeatureEmbeddingModule(nsample, radius, mlp[-1]+7, [mlp[-1] for i in range(len(mlp))])
        # self.sample = SampleLayer(npoint, mlp[-1]+3)
        self.conv3 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.up0=ME.MinkowskiConvolutionTranspose(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.conv1=ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.up1=ME.MinkowskiConvolutionTranspose(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.conv2=ME.MinkowskiConvolution(
            64,
            3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.up2=ME.MinkowskiConvolutionTranspose(
            128,
            64,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3
        )
        self.TCM=TCM(128)
        self.GCM=GCM(128)
        self.TCM1=TCM(128)
        self.GCM1=GCM(128)
        # self.bn1=ME.MinkowskiBatchNorm(128)
        # self.bn2=ME.MinkowskiBatchNorm(128)
        # self.bn3=ME.MinkowskiBatchNorm(128)

        # for p in self.parameters():
        #     p.requires_grad=False
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x=self.conv3(x)
        x=(self.GCM1(self.TCM1(x)))
        x=self.GCM(self.TCM((self.relu((self.conv0(self.up0(x)))))))
        x=self.relu((self.conv1(self.up1((x)))))
        x=(self.conv2(self.up2((x))))


        return x

class Hyper_encoder(nn.Module):
    def __init__(self):
        super(Hyper_encoder, self).__init__()
        self.TCM1 = TCM(128)
        self.GCM1 = GCM(128)
        self.conv0 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.bn1=ME.MinkowskiBatchNorm(128)
        # self.bn2=ME.MinkowskiBatchNorm(128)


    def forward(self, x):
        x=self.relu((self.conv0(x)))
        x=self.relu((self.down0(self.conv1(x))))
        x=self.relu(self.GCM1(self.TCM1(x)))

        x=(self.down1(self.conv2(x)))

        return x

class Hyper_decoder(nn.Module):
    def __init__(self):
        super(Hyper_decoder, self).__init__()
        self.TCM1 = TCM(128)
        self.GCM1 = GCM(128)
        self.conv0 = ME.MinkowskiConvolution(
            256,
            256,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            256,
            256,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.up0 = ME.MinkowskiConvolutionTranspose(
            128,
            256,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            128,
            128,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.up1 = ME.MinkowskiConvolutionTranspose(
            128,
            128,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.bn1=ME.MinkowskiBatchNorm(128)
        # self.bn2=ME.MinkowskiBatchNorm(256)

    def forward(self, x):
        x=self.relu((self.conv2(self.up1(x))))
        x=self.GCM1(self.TCM1(x))
        x=self.relu((self.conv1(self.up0(x))))
        x=self.conv0(x)

        return x



class TCM(nn.Module):
    def __init__(self,input_dim):
        super(TCM, self).__init__()
        self.dim=input_dim
        self.conv1=ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.trans=CodedVTRBlock(input_dim//2,input_dim//2)
        # self.trans=resblock_dia(input_dim//2,input_dim//2,1)

        self.resblock1=resblock(input_dim//2,input_dim//2,1)
    def forward(self,x):
        conv_x,trans_x=torch.split(self.conv1(x).F,(self.dim//2,self.dim//2),dim=1)
        conv_x=ME.SparseTensor(
                features=conv_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        conv_x=self.resblock1(conv_x)+conv_x
        trans_x=ME.SparseTensor(
                features=trans_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        trans_x=self.trans(trans_x)
        res=self.conv2(ME.cat(conv_x,trans_x))
        x=x+res
        return x
class GCM(nn.Module):
    def __init__(self,input_dim):
        super(GCM, self).__init__()
        self.dim=input_dim
        self.conv1=ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.glob=glob_feat(input_dim//2)
        self.resblock1=resblock(input_dim//2,input_dim//2,1)

    def forward(self,x):
        conv_x,glo_x=torch.split(self.conv1(x).F,(self.dim//2,self.dim//2),dim=1)
        conv_x=ME.SparseTensor(
                features=conv_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        conv_x=self.resblock1(conv_x)+conv_x
        glo_x = ME.SparseTensor(
            features=glo_x,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)
        glo_x=self.glob(glo_x)

        res=self.conv2(ME.cat(conv_x,glo_x))
        x=x+res
        return x



class resblock(nn.Module):
    def __init__(self,input_dim,output_dim,stride):
        super(resblock, self).__init__()
        self.conv1=ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=stride,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=1,
            stride=stride,
            bias=True,
            dimension=3)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out

class resblock_dia(nn.Module):
    def __init__(self,input_dim,output_dim,stride):
        super(resblock_dia, self).__init__()
        self.conv1=ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=3,
            dilation=2,
            stride=stride,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            dilation=2,

            stride=1,
            bias=True,
            dimension=3)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=1,
            dilation=2,
            stride=stride,
            bias=True,
            dimension=3)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out

class resblock_up(nn.Module):
    def __init__(self,input_dim,output_dim,stride):
        super(resblock_up, self).__init__()
        self.conv1=ME.MinkowskiConvolutionTranspose(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=stride,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=1,
            stride=stride,
            bias=True,
            dimension=3)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out
def ME_conv(inc,outc,kernel_size=3,stride=1):
    return ME.MinkowskiConvolution(inc,outc,kernel_size=kernel_size,stride=stride,bias=True,dimension=3)

def ste_round(x):
    return torch.round(x)-x.detach()+x
class glob_feat(nn.Module):
    def __init__(self,dim):
        super(glob_feat, self).__init__()
        self.conv1=ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2=ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv3=ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
    def forward(self,x):
        # out1_space=self.relu(self.conv1(x)).F.mean(dim=-1,keepdim=True)
        # out2_channel=self.relu(self.conv2(x)).F.mean(dim=0,keepdim=True)
        # final_encoding = torch.matmul(out1_space, out2_channel)+out1_space+out2_channel
        # final_encoding = torch.sqrt(final_encoding+1e-12) # B,C/8,N
        # final_encoding=ME.SparseTensor(
        #     features=final_encoding,
        #     coordinate_map_key=x.coordinate_map_key,
        #     coordinate_manager=x.coordinate_manager,
        #     device=x.device)
        # final_encoding =self.relu(self.conv3(final_encoding))
        # final_encoding=x-final_encoding#zui hou meijia relu
        # return final_encoding
        # out1_space=self.relu(self.conv1(x)).F.mean(dim=-1,keepdim=True)
        # out2_channel=self.relu(self.conv2(x)).F.mean(dim=0,keepdim=True)
        out1_space=self.relu(self.conv1(x))
        out2_channel=self.relu(self.conv2(x))
        all=[]
        feats1=out1_space.decomposed_features
        feats2=out2_channel.decomposed_features

        for i in range(len(feats1)):
            final_encoding = torch.matmul(feats1[i].mean(dim=-1,keepdim=True), feats2[i].mean(dim=0,keepdim=True))
            final_encoding = torch.sqrt(final_encoding+1e-12) # B,C/8,N
            all.append(final_encoding)
        final_encoding=ME.SparseTensor(
            features=torch.cat(all,dim=0)+out1_space.F+out2_channel.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)
        final_encoding =self.relu(self.conv3(final_encoding))
        final_encoding=x-final_encoding#zui hou meijia relu
        return self.relu(final_encoding)




