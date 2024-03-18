# TSC-PCAC
Point Cloud Attribute Compression with Sparse Convolution and Voxel Transformer

## Requirments

- CUDA=11.1

- pytorch=1.9.0

- python=3.8

- minkowskiengine=0.5.4

- torchac=0.9.3

- preweight models: <https://drive.google.com/file/d/1wzfq6fwUkKoe46oxzCYmRrUDPgLAO-Zd/view?usp=drive_link>

## Test

python codec.py --dataset_path test_rootdir

## Train

python train.py --batch_size 8 --learning 1e-4 --lamb 16000 --dataset_path train_rootdir --val_path val_rootdir
