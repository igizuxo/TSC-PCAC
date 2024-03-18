
import argparse
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
import logging
from pathlib import Path
import sys
import os, sys, glob
import importlib
from dataset import *
from torch import nn
import torch.nn.init as init
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from torch.utils.tensorboard import SummaryWriter
import MinkowskiEngine as ME
import gc
import random
from torch.utils.data.distributed import DistributedSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('TSC-PCAC')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='TSC-PCAC', help='model name')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device [default: 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=2e-5, help='decay rate [default: 1e-4]')
    parser.add_argument('--lamb', default=16000, type=int)
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--multigpu', default=False, type=bool)
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--val_path', type=str, default='')

    return parser.parse_args()


def traverse_path_recursively(rootdir):
        filedirs = []

        def gci(filepath):
            files = os.listdir(filepath)
            for fi in files:
                fi_d = os.path.join(filepath, fi)
                if os.path.isdir(fi_d):
                    gci(fi_d)
                else:
                    filedirs.append(os.path.join(fi_d))
            return

        gci(rootdir)

        return filedirs
def test( model, loader, criterion, writer,epoch):
    mean_loss = []
    mean_bpp = []
    mean_mse = []
    length = len(loader)
    with torch.no_grad():
        for j, data in enumerate(loader):



            xyz = data[0]
            points = data[1]
            x = ME.SparseTensor(features=points, coordinates=xyz, device=device)


            model.eval()
            bpp, pc_coor, mse = model(x)
            loss, mse, bpp = criterion(bpp, mse)
            mean_mse.append(mse.mean().item())
            mean_loss.append(loss.mean().item())
            mean_bpp.append(bpp.mean().item())
            # writer.
            writer.add_scalars(main_tag='val/losses',
                               tag_scalar_dict={'sum_loss': loss},
                               global_step=epoch)

            writer.add_scalars(main_tag='val/metrics',
                               tag_scalar_dict={
                                   'mbpp': (bpp)},
                               global_step=epoch)
            writer.add_scalars(main_tag='val/metrics',
                               tag_scalar_dict={
                                   'mmse': mse},
                               global_step=epoch)
    return np.mean(mean_loss), np.mean(mean_bpp), np.mean(mean_mse)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    if args.multigpu:
        # seed = 42
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
    device = torch.device('cuda', args.local_rank)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_name = str(args.lamb)
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(experiment_name)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    if args.multigpu:
        if args.local_rank == 0:
            writer = SummaryWriter(log_dir=log_dir)
    elif args.multigpu==False:
        writer = SummaryWriter(log_dir=log_dir)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if args.local_rank==0:
        log_string('PARAMETER ...')
        log_string(args)

        '''DATA LOADING'''
        log_string('Load dataset ...')

    filedirs_train = args.dataset_path
    input_filedirs = traverse_path_recursively(rootdir=filedirs_train)
    filedirs_train = [f for f in input_filedirs if
                     (os.path.splitext(f)[1] == '.h5' or os.path.splitext(f)[1] == '.ply')]  # .off or .obj
    log_string(len(filedirs_train))

    filedirs_val = sorted(glob.glob(args.val_path +'/*.ply'))

    TRAIN_DATASET = PCDataset(filedirs_train
                            )
    VAL_DATASET = PCDataset(filedirs_val
                          )
    if args.multigpu:
        train_sampler = DistributedSampler(TRAIN_DATASET)
        trainDataLoader = make_data_loader_mulgpu(dataset=TRAIN_DATASET, train_sampler=train_sampler,batch_size=args.batch_size, shuffle=True,
                                           num_workers=4, repeat=False,
                                           collate_fn=collate_pointcloud_fn)
        val_sampler = DistributedSampler(VAL_DATASET)

        valDataLoader = make_data_loader_mulgpu(dataset=VAL_DATASET,train_sampler=val_sampler, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                         repeat=False,
                                         collate_fn=collate_pointcloud_fn)
    else:
        trainDataLoader = make_data_loader(dataset=TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, repeat=False,
                                       collate_fn=collate_pointcloud_fn)
        valDataLoader = make_data_loader(dataset=VAL_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=1, repeat=False,
                                       collate_fn=collate_pointcloud_fn)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)

    if not args.cpu:
        model = MODEL.get_model(
                               ).to(device)
        criterion = MODEL.get_loss(lam=args.lamb).to(device)
    else:
        model = MODEL.get_model(
                               )
        criterion = MODEL.get_loss(lam=args.lamb)

    '''pretrain or train from scratch'''
    try:
        if args.multigpu:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            pretrained_dict={key.replace('module.',''): value for key,value in checkpoint['model_state_dict'].items()}
            new_dict = model.state_dict()
            a = new_dict
            for k, v in pretrained_dict.items():
                if k in new_dict:
                    new_dict[k] = v
            model.load_state_dict(new_dict)
            if args.local_rank == 0:

                log_string('Use pretrain model')
        else:

            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cpu'))
            start_epoch = checkpoint['epoch']
            pretrained_dict={key.replace('module.',''): value for key,value in checkpoint['model_state_dict'].items()}
            #在新的模型中加载原来已有模型的参数
            new_dict = model.state_dict()
            a=new_dict
            for k, v in pretrained_dict.items():
                if k in new_dict:
                    new_dict[k] = v

            model.load_state_dict(new_dict)
            if args.local_rank == 0:

                log_string('Use pretrain model')
    except:
        if args.local_rank == 0:

            log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.multigpu:
        if args.local_rank == 0:

            print('multiple gpu used')
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        if args.local_rank == 0:

            log_string('Use pretrain optimizer')
    try:
        assert len(args.pretrained_model) == 0
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.local_rank==0:
            print(optimizer)
    except:
        if args.local_rank == 0:

            log_string('No existing optimizer')
    global_step = 0
    if args.log_dir:
        best_loss_test = checkpoint['loss']
    else:
        best_loss_test = 9999999

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        # train_sampler.set_epoch(epoch)
        mean_loss = []
        mean_bpp_loss = []
        mean_mse_loss = []
        if args.local_rank == 0:

            log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))

        #adjust lf
        log_string(optimizer.param_groups[0]['lr'])
        for batch_id, x in enumerate(tqdm(trainDataLoader)):
            # points = data[:, :, 3:6]
            # xyz = data[:, :, 0:3]
            xyz=x[0]
            points=x[1]
            x = ME.SparseTensor(features=points, coordinates=xyz, device=device)
            optimizer.zero_grad()
            model.train()
            bpp, pc_coor, mse = model(x)
            loss, mse, bpp = criterion(bpp, mse)
            if args.multigpu:
                loss = loss.mean()
                mse = mse.mean()
                bpp = bpp.mean()

            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())
            mean_bpp_loss.append(bpp.item())
            mean_mse_loss.append(mse.item())
            global_step += 1
            del bpp, pc_coor, mse, x,xyz,points
            torch.cuda.empty_cache()
            gc.collect()
        ml = np.mean(mean_loss)
        mbpp = np.mean(mean_bpp_loss)
        mmse = np.mean(mean_mse_loss)
        if args.multigpu and args.local_rank==0:
            log_string('1024*Batch point mean loss: %f' % ml)
            log_string('1024*Batch point mean bpp: %f' % mbpp)
            log_string('1024*Batch point MSE: %f' % mmse)
            # writer.
            writer.add_scalars(main_tag='train/losses',
                               tag_scalar_dict={'sum_loss': ml},
                               global_step=epoch)

            writer.add_scalars(main_tag='train/metrics',
                               tag_scalar_dict={
                                   'bpp': (mbpp)
                               },
                               global_step=epoch)
            writer.add_scalars(main_tag='train/metrics',
                               tag_scalar_dict={

                                   'mmse': mmse},
                               global_step=epoch)

        elif args.multigpu==False:
            log_string('1024*Batch point mean loss: %f' % ml)
            log_string('1024*Batch point mean bpp: %f' % mbpp)
            log_string('1024*Batch point MSE: %f' % mmse)
            # writer.
            writer.add_scalars(main_tag='train/losses',
                               tag_scalar_dict={'sum_loss': ml},
                               global_step=epoch)

            writer.add_scalars(main_tag='train/metrics',
                               tag_scalar_dict={
                                   'bpp': (mbpp)
                               },
                               global_step=epoch)
            writer.add_scalars(main_tag='train/metrics',
                               tag_scalar_dict={

                                   'mmse': mmse},
                               global_step=epoch)
        # scheduler.step(ml)
        if epoch % 1 == 0 and args.local_rank==0:
            log_string('Start val...')
            with torch.no_grad():
                mean_loss_test, mean_bpp_test, mean_mse_test = test( model.eval(), valDataLoader, criterion,
                                                                   writer,epoch)
                log_string('val loss: %f' % (mean_loss_test))
                log_string('val bpp: %f' % (mean_bpp_test))
                log_string('val mse: %f' % (mean_mse_test))

                if epoch % 10 == 0:
                    savepath = str(checkpoints_dir) + '/' + str(epoch) + '.pth'
                    state = {
                        'epoch': epoch,
                        'loss': mean_loss_test,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)

                if (mean_loss_test < best_loss_test and epoch >= 200):
                    logger.info('Save model...')
                    best_loss_test = mean_loss_test
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'loss': mean_loss_test,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
    if args.local_rank==0:
        writer.close()
        logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
