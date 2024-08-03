##!/usr/bin/env bash

SAVE_DIR="./output/tank/Effi_MVS_plus"

TANK_TESTING='/data1/local_userdata/wangshaoqian/TankandTemples/TankandTemples/'

CKPT_FILE="./checkpoints/Effi_MVS_plus/model_tank.ckpt"


if [ ! -d $SAVE_DIR ]; then
    mkdir -p $SAVE_DIR
fi
#
CUDA_VISIBLE_DEVICES=0 python test_tank.py --dataset=tank --batch_size=1 --testpath=$TANK_TESTING --GRUiters 3,3,3 --ndepths 96,8,8 --CostNum=3 --numdepth=384 --initloss=initloss --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --savedir $SAVE_DIR --outdir $SAVE_DIR --data_type tank \
              --num_view=11 --dispmaxfirst=last



