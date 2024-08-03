#!/usr/bin/env bash


MVS_TRAINING="/home/wangshaoqian/data/DTU/mvs_training/dtu/"

BLEND_TRAINING="/data1/local_userdata/wangshaoqian/dataset_low_res/"

LOG_DIR="./checkpoints/Effi_MVS_plus"
LOG_DIR_CKPT="./checkpoints/Effi_MVS_plus/model_dtu.ckpt"
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

filename="Effi_MVS_plus"
dirAndName="$LOG_DIR/$filename.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

#train

CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py --mode='train' --epochs=16 --numdepth=384 --trainviews=5 --testviews=5 --logdir $LOG_DIR --dataset=dtu_yao --batch_size=16 --trainpath=$MVS_TRAINING \
                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --GRUiters 3,3,3 --ndepths 48,8,8 --CostNum=3 --initloss=initloss --lr=0.001 --dispmaxfirst=last  --maskupmode=eachstage --interval_scale=0.53 --lossrate=0.9 | tee -i $dirAndName
#finetune
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -u train.py --mode='finetune' --epochs=10 --numdepth=384 --trainviews=7 --testviews=7 --logdir $LOG_DIR --loadckpt $LOG_DIR_CKPT --dataset=blend --batch_size=8 --trainpath=$BLEND_TRAINING \
#                --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --ndepths=96 --CostNum=3 --initloss=initloss --lr=0.0004 --dispmaxfirst=last  --maskupmode=eachstage --interval_scale=0.53  --lossrate=0.9 | tee -i $dirAndName
