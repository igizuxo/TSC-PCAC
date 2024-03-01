
from models.utils import *
from models.bitEstimator import BitEstimator
import math
import numpy as np
import torchac
import gc
import os

def read_binary_files(filename, rootdir='./output'):
    """Read from compressed binary files:
      1) Compressed latent features.
      2) Compressed hyperprior.
      3) Number of input points.
    """

    print('===== Read binary files =====')
    file_strings = os.path.join(rootdir, filename + '.strings')
    file_strings_hyper = os.path.join(rootdir, filename + '.strings_hyper')

    with open(file_strings, 'rb') as f:
        y_shape = np.frombuffer(f.read(2*2), dtype=np.int16)
        y_min_v, y_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
        y_strings = f.read()

    with open(file_strings_hyper, 'rb') as f:
        z_shape = np.frombuffer(f.read(2*2), dtype=np.int16)
        z_min_v, z_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
        z_strings = f.read()

    return y_strings, z_strings, y_min_v, y_max_v, y_shape, z_min_v, z_max_v, z_shape
class get_model(nn.Module):
    def __init__(self):#npoint 为原始点数
        super(get_model, self).__init__()
        self.encoder=enconder()
        self.decoder=deconder()
        self.hyer_encoder=Hyper_encoder()
        self.hyer_decoder=Hyper_decoder()
        self.bitEstimator_z=BitEstimator(128)
        self.use_hyper=True
        self.crt=torch.nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)
        self.num_slices = 8
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.mu_transform = nn.ModuleList(
            nn.Sequential(
                ME_conv(128 + 128//self.num_slices * (i), 32),
                ME.MinkowskiReLU(inplace=True),
                TCM(32),
                GCM(32),
                ME_conv(32, 128//self.num_slices),

            ) for i in range(self.num_slices)
        )
        self.scale_transform = nn.ModuleList(
            nn.Sequential(
                ME_conv(128 + 128//self.num_slices * (i), 32),
                ME.MinkowskiReLU(inplace=True),
                TCM(32),
                GCM(32),
                ME_conv(32, 128//self.num_slices),
            ) for i in range(self.num_slices)
        )
        self.lrp = nn.ModuleList(
            nn.Sequential(
                ME_conv(128 +128//self.num_slices * (i+1), 32),
                ME.MinkowskiReLU(inplace=True),
                TCM(32),
                GCM(32),
                ME_conv(32, 128//self.num_slices),
            ) for i in range(self.num_slices)
        )


    def _likelihood_z(self, z):
        likelihood=self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        return likelihood
    def compress(self, x):
        values = x
        xshape=x.shape
        min_v, max_v = values.min(), values.max()
        symbols = torch.arange(min_v, max_v+1).cuda()
        symbols = symbols.reshape(1, -1).repeat( values.shape[-1],1)
        values_norm = (values - min_v).type(torch.int16)
        pmf = self._likelihood_z(symbols.T)
        pmf=pmf.T
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True)) #dim=K de dim
        cdf = self._pmf_to_cdf(pmf)
        cdf=cdf.clamp(max=1.).cpu()
        cdf=cdf.repeat( values.shape[0],1,1)
        strings = torchac.encode_float_cdf(cdf, values_norm.cpu(), check_input_bounds=True)
        return strings, min_v, max_v,xshape

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        if len(pmf.shape)==3:
            cdf = torch.cat((torch.zeros_like(cdf[:,:, :1]), cdf), dim=-1)
        else:
            cdf = torch.cat((torch.zeros_like(cdf[:,:1]), cdf), dim=-1)

        return cdf
    def encode(self,x,filename):
        y = self.encoder(x)
        z = self.hyer_encoder(y)
        compressed_z = torch.round(z.F)
        z_string, zmin, zmax, z_shape = self.compress(compressed_z)
        compressed_z = ME.SparseTensor(
            features=compressed_z,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)
        hyer = self.hyer_decoder(compressed_z)
        xslice = y.F.chunk(self.num_slices, 1)
        compressed_y = []
        rootdir='./output'
        if not os.path.exists('./output'):
            os.makedirs('./output')
        file_strings='./output/'+filename+'.strings'
        y_hat_slices = []
        mu_all = []
        scale_all = []
        for index, y_slice in enumerate(xslice):
            support_slices = (y_hat_slices)  # tensor list

            mu_support = torch.cat([hyer.F[:, :128]] + support_slices, dim=1)
            scale_support = torch.cat([hyer.F[:, 128:]] + support_slices, dim=1)

            mu_support = ME.SparseTensor(features=mu_support,
                                         coordinate_map_key=y.coordinate_map_key,
                                         coordinate_manager=y.coordinate_manager,
                                         device=y.device)
            scale_support = ME.SparseTensor(features=scale_support,
                                            coordinate_map_key=y.coordinate_map_key,
                                            coordinate_manager=y.coordinate_manager,
                                            device=y.device)
            mu = self.mu_transform[index](mu_support).F
            mu_all.append(mu)
            scale =torch.exp(self.scale_transform[index](scale_support).F).clamp(1e-10, 1e10)
            scale_all.append(scale)
            y_hat_slice = ste_round(y_slice)
            compressed_y.append(y_hat_slice.clone())
            lrp_support = torch.cat([mu_support.F, y_hat_slice], dim=1)
            lrp_support = ME.SparseTensor(features=lrp_support,
                                          coordinate_map_key=y.coordinate_map_key,
                                          coordinate_manager=y.coordinate_manager,
                                          device=y.device)
            lrp = self.lrp[index](lrp_support).F
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)



        compressed_y=torch.cat(compressed_y, dim=0)
        mu=torch.cat(mu_all, dim=0)
        scale=torch.cat(scale_all, dim=0)
        ymin, ymax = compressed_y.min(), compressed_y.max()
        symbols = torch.arange(ymin, ymax + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(128//self.num_slices , 1).repeat(len(mu), 1, 1)
        compressed_y = (compressed_y - ymin).type(torch.int16).cpu()
        if len(mu.shape) == 2:
            mu = mu.unsqueeze(2)
            scale = scale.unsqueeze(2)

        # pmf
        pmf = 0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
              - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale))
        torch.cuda.empty_cache()
        gc.collect()
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        # get CDF
        cdf = (self._pmf_to_cdf(pmf))
        cdf = cdf.clamp(max=1.).cpu()
        torch.cuda.empty_cache()
        gc.collect()
        strings = torchac.encode_float_cdf(cdf, compressed_y, check_input_bounds=True)
        file_strings_hyper = os.path.join(rootdir, filename + '.strings_hyper')
        with open(file_strings_hyper, 'wb') as f:
            f.write(np.array(z_shape, dtype=np.int16).tobytes())  # [batch size, length, width, height, channels]
            f.write(np.array((zmin.cpu(), zmax.cpu()), dtype=np.int8).tobytes())
            f.write(np.array((ymin.cpu(), ymax.cpu()), dtype=np.int8).tobytes())
            f.write(z_string)
        with open(file_strings, 'wb') as f:
            f.write(strings)
        bytes_strings_hyper = os.path.getsize(file_strings_hyper)
        bytes_strings = os.path.getsize(file_strings)
        print('Total file size (Bytes): {}'.format(bytes_strings + bytes_strings_hyper))
        print('Strings (Bytes): {}'.format(bytes_strings))
        print('Strings hyper (Bytes): {}'.format(bytes_strings_hyper))
        return bytes_strings, bytes_strings_hyper,compressed_z
    def decompress(self, strings, min_v, max_v, shape):
        symbols = torch.arange(min_v, max_v+1)
        symbols = torch.tensor(symbols.reshape(1, -1).repeat( shape[-1],1)).cuda()
        pmf = self._likelihood_z(symbols.T)
        pmf=pmf.T
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True)) #dim=K de dim
        cdf = self._pmf_to_cdf(pmf)
        cdf=cdf.clamp(max=1.).cpu()
        cdf=cdf.repeat(shape[0],1,1)
        values = torchac.decode_float_cdf(cdf, strings)
        values+=min_v
        return values.float()
    def decode(self,filename,compressed_z):
        print('===== Read binary files =====')
        rootdir='output'
        file_strings = os.path.join(rootdir, filename + '.strings')
        file_strings_hyper = os.path.join(rootdir, filename + '.strings_hyper')
        with open(file_strings_hyper, 'rb') as f:
            z_shape = np.frombuffer(f.read(2 * 2), dtype=np.int16)
            z_min_v, z_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
            y_min_v, y_max_v = np.frombuffer(f.read(2 * 1), dtype=np.int8)
            z_strings = f.read()
        with open(file_strings, 'rb') as f:
            y_strings = f.read()
        z = self.decompress(z_strings, z_min_v, z_max_v, z_shape)
        compressed_z = ME.SparseTensor(
            features=z,
            coordinate_map_key=compressed_z.coordinate_map_key,
            coordinate_manager=compressed_z.coordinate_manager,
            device=compressed_z.device)
        hyer = self.hyer_decoder(compressed_z)
        y_hat_slices = []
        #init
        mu_all=torch.zeros((len(hyer)*self.num_slices,128//self.num_slices)).cuda()
        scale_all=(torch.zeros((len(hyer)*self.num_slices,128//self.num_slices))+0.01).cuda()

        symbols = torch.arange(y_min_v, y_max_v + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(128 // self.num_slices, 1).repeat(len(mu_all), 1, 1)
        if len(mu_all.shape) == 2:
            mu = mu_all.unsqueeze(2)
            scale = scale_all.unsqueeze(2)
        del mu_all,scale_all
        torch.cuda.empty_cache()
        gc.collect()

        pmf = (0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
              - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale)))
        pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
        # get CDF
        cdf = (self._pmf_to_cdf(pmf)).clamp(max=1.).cpu()
        torch.cuda.empty_cache()
        gc.collect()
        symbols = torch.arange(y_min_v, y_max_v + 1).cuda()
        symbols = symbols.reshape(1, -1).repeat(128 // self.num_slices, 1).repeat(len(hyer), 1, 1)
        for index in range(self.num_slices):
            support_slices = (y_hat_slices)  # tensor list

            mu_support = torch.cat([hyer.F[:, :128]] + support_slices, dim=1)
            scale_support = torch.cat([hyer.F[:, 128:]] + support_slices, dim=1)

            mu_support = ME.SparseTensor(features=mu_support,
                                         coordinate_map_key=hyer.coordinate_map_key,
                                         coordinate_manager=hyer.coordinate_manager,
                                         device=hyer.device)
            scale_support = ME.SparseTensor(features=scale_support,
                                            coordinate_map_key=hyer.coordinate_map_key,
                                            coordinate_manager=hyer.coordinate_manager,
                                            device=hyer.device)


            mu_all = self.mu_transform[index](mu_support).F
            scale_all =torch.exp(self.scale_transform[index](scale_support).F).clamp(1e-10, 1e10)
            # pmf
            if len(mu_all.shape) == 2:
                mu = mu_all.unsqueeze(2)
                scale = scale_all.unsqueeze(2)
            pmf = 0.5 - 0.5 * (symbols + 0.5 - mu).sign() * torch.expm1(-(symbols + 0.5 - mu).abs() / scale) \
                  - (0.5 - 0.5 * (symbols - 0.5 - mu).sign() * torch.expm1(-(symbols - 0.5 - mu).abs() / scale))
            # torch.cuda.empty_cache()
            # gc.collect()
            pmf = pmf / (pmf.sum(dim=-1, keepdim=True))  # dim=K de dim
            # get CDF
            if index==(self.num_slices-1):
                a=(self._pmf_to_cdf(pmf)).clamp(max=1.).cpu()
                cdf[index*len(hyer):,:,:] = (self._pmf_to_cdf(pmf)).clamp(max=1.).cpu()
            else:
                a=(self._pmf_to_cdf(pmf)).clamp(max=1.).cpu()
                cdf[index*len(hyer):(index+1)*len(hyer),:,:] = (self._pmf_to_cdf(pmf)).clamp(max=1.).cpu()


            # decoding
            if index==(self.num_slices-1):
                y_hat_slice=((torchac.decode_float_cdf(cdf, y_strings))[index*len(hyer):,:]+y_min_v).cuda().float()
            else:
                y_hat_slice=((torchac.decode_float_cdf(cdf, y_strings))[index*len(hyer):(index+1)*len(hyer),:]+y_min_v).cuda().float()
            lrp_support = torch.cat([mu_support.F, y_hat_slice], dim=1)
            lrp_support = ME.SparseTensor(features=lrp_support,
                                          coordinate_map_key=hyer.coordinate_map_key,
                                          coordinate_manager=hyer.coordinate_manager,
                                          device=hyer.device)
            lrp = self.lrp[index](lrp_support).F
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
            torch.cuda.empty_cache()
            gc.collect()
        y_hat_slices = torch.cat(y_hat_slices, dim=1)
        y_hat_slices = ME.SparseTensor(features=y_hat_slices,
                                       coordinate_map_key=hyer.coordinate_map_key,
                                       coordinate_manager=hyer.coordinate_manager,
                                       device=hyer.device)
        rec_x=self.decoder(y_hat_slices)
        return rec_x
    def forward(self, x):
        pc_gd=x.F
        x=self.encoder(x)

        if self.use_hyper:
            z = self.hyer_encoder(x)
            z_noise = torch.nn.init.uniform_(torch.zeros_like(z.F), -0.5, 0.5)
            if self.training:
                compressed_z = z.F + z_noise
            else:
                compressed_z = torch.round(z.F)
        compressed_z = ME.SparseTensor(
            features=compressed_z,
            coordinate_map_key=z.coordinate_map_key,
            coordinate_manager=z.coordinate_manager,
            device=z.device)
        # decoder
        if self.use_hyper:
            y_hat_slices = []
            mu_all=[]
            scale_all=[]
            hyer = self.hyer_decoder(compressed_z)
            xslice = x.F.chunk(self.num_slices, 1)
            for index, y_slice in enumerate(xslice):
                support_slices = (y_hat_slices)  # tensor list

                mu_support = torch.cat([hyer.F[:,:128]] + support_slices, dim=1)
                scale_support = torch.cat([hyer.F[:,128:]] + support_slices, dim=1)

                mu_support = ME.SparseTensor(features=mu_support,
                                          coordinate_map_key=x.coordinate_map_key,
                                          coordinate_manager=x.coordinate_manager,
                                          device=x.device)
                scale_support = ME.SparseTensor(features=scale_support,
                                             coordinate_map_key=x.coordinate_map_key,
                                             coordinate_manager=x.coordinate_manager,
                                             device=x.device)
                mu = self.mu_transform[index](mu_support).F
                scale = self.scale_transform[index](scale_support).F
                mu_all.append(mu)
                scale_all.append(scale)
                y_hat_slice=ste_round(y_slice-mu)+mu

                lrp_support=torch.cat([mu_support.F,y_hat_slice],dim=1)
                lrp_support = ME.SparseTensor(features=lrp_support,
                                                coordinate_map_key=x.coordinate_map_key,
                                                coordinate_manager=x.coordinate_manager,
                                                device=x.device)
                lrp = self.lrp[index](lrp_support).F
                lrp =0.5*torch.tanh(lrp)
                y_hat_slice+=lrp

                y_hat_slices.append(y_hat_slice)
            mu_all = torch.cat(mu_all, dim=1)
            scale_all = torch.cat(scale_all, dim=1)
            y_hat_slices = torch.cat(y_hat_slices, dim=1)
            y_hat_slices = ME.SparseTensor(features=y_hat_slices,
                                            coordinate_map_key=x.coordinate_map_key,
                                            coordinate_manager=x.coordinate_manager,
                                            device=x.device)
        attr_recon = self.decoder(y_hat_slices)
        if self.training:
            x_noise=torch.nn.init.uniform_(torch.zeros_like(x.F),-0.5,0.5)
            compressed_x=x.F+x_noise
        else:
            compressed_x=torch.round(x.F)
        def feature_probs_based_sigma_test(feature,mu,scale):
            sigma = torch.exp(scale).clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob



        total_bits_feature, _ = feature_probs_based_sigma_test(compressed_x, mu_all,scale_all)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z.F)
        bpp = (total_bits_z + total_bits_feature) / (len(pc_gd))

        MSE=self.crt(pc_gd,attr_recon.F)

        return bpp, attr_recon, MSE

class get_loss(nn.Module):
    def __init__(self, lam=1):
        super(get_loss, self).__init__()
        self.lam = lam

    def forward(self, bit, mse):
        return self.lam*mse + bit, mse, bit
if __name__ == '__main__':
    model = get_model()
    num_para=sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Number size: {num_para}")
