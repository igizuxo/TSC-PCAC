# TSC-PCAC
Point Cloud Attribute Compression with Sparse Convolution and Voxel Transformer

## Requirments

CUDA=11.1

pytorch=1.9.0

python=3.8

minkowskiengine==0.5.4

torchac==0.9.3

## Test

python codec.py --dataset_path test_path

## Train

python train.py --batch_size 8 --learning 1e-4 --lamb 16000 --dataset_path train_path --val_path val_path
