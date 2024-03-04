
import collections
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import open3d
import re
import pandas as pd
from dataset import *
import time
import importlib
import sys
import argparse
import gc
from utils1.pc_error_wrapper import pc_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('TSC-PCAC')
    parser.add_argument('--dataset_path', type=str, default='')
    return parser.parse_args()
if __name__ == '__main__':


    args = parse_args()
    ckpt_of_different_rates = ['r0', 'r1', 'r2','r3', 'r4'
                               ]
    ckpt_of_different_rates=ckpt_of_different_rates[::-1]
    filedirs_train = sorted(glob.glob(args.dataset_path + '/*.ply'))
    test_data = PCDataset(filedirs_train)
    test_loader = make_data_loader(test_data, num_workers=1, batch_size=1, shuffle=False)
    outdir = './output'

    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(outdir): os.makedirs(outdir)

    idx=0
    idx_rate=0
    for exp_name in ckpt_of_different_rates:
        model_name = 'TSC-PCAC'
        experiment_dir = 'ckpts/' + exp_name + '.pth'
        MODEL = importlib.import_module(model_name)
        model = MODEL.get_model().cuda()
        model.eval()
        checkpoint = torch.load(str(experiment_dir))
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint.items():
            if '_map' not in k:
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print('=' * 10, 'Test', '=' * 10)
        for step, data in enumerate(test_loader):
            with torch.no_grad():
                pc_data = data[1]
                xyz = data[0]
                x = ME.SparseTensor(features=pc_data, coordinates=xyz, device=device)
                filedir=test_data.files[step]
                filename = os.path.split(filedir)[-1].split('.')[0]
                print(filename)
                # encode
                start_time = time.time()
                bytes_strings,bytes_strings_hyper,compress_z = model.encode(x,filename)
                enc_time=round(time.time() - start_time, 3)

                print('Enc Time:\t',enc_time, 's')
                torch.cuda.empty_cache()
                gc.collect()

                # decode
                start_time = time.time()
                #  only use coor of compress_z instead of feature for decode since coor is known in decoder
                x_dec = model.decode(filename,compress_z)
                dec_time=round(time.time() - start_time, 3)
                print('Dec Time:\t', dec_time, 's')
                bpp=8*(bytes_strings+bytes_strings_hyper)/(len(xyz))
                rec_pcd = open3d.geometry.PointCloud() 
                rec_pcd.points = open3d.utility.Vector3dVector((x_dec.C[:, 1:].cpu()).float()) 
                rec_pcd.colors = open3d.utility.Vector3dVector(yuv_rgb((x_dec.F.cpu())))  
                if not os.path.exists('pc_file'): os.makedirs('pc_file')
                recfile = 'pc_file/' + filename + '_r' + str(idx_rate) + '_rec.ply'
                open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
                pc_error_metrics = pc_error(infile1=filedir, infile2=recfile, res=1) 
                pc_errors = [pc_error_metrics["c[0],PSNRF"][0],
                             pc_error_metrics["c[1],PSNRF"][0],
                             pc_error_metrics["c[2],PSNRF"][0]]

                results = pc_error_metrics
                results["bpp"] = np.array(bpp).astype('float32')
                results["enc_time"] = np.array((enc_time)).astype('float32')
                results["dec_time"] = np.array(dec_time)
                fimename=os.path.split(str(test_data.files[step]))[-1].split('.')[0]
                fimename=re.split(r'\d+',fimename)[0]
                results['sequence'] = fimename
                last_col = results.pop(results.columns[-1])
                results.insert(0, last_col.name, last_col)
                if step == 0:
                    all_result = results.copy(deep=True)
                else:
                    all_result = all_result.append(results, ignore_index=True)
                torch.cuda.empty_cache()
                gc.collect()
        all_result=all_result.groupby('sequence').mean().reset_index()
        idx_rate+=1
        if idx == 0:
            global all_results
            all_results = all_result.copy(deep=True)
            idx = 1
        else:
            all_results = all_results.append(all_result, ignore_index=True)
    if not os.path.exists('result'): os.makedirs('result')
    csv_name = os.path.join('./result/', 'result.csv')
    all_results.to_csv(csv_name, index=False)




