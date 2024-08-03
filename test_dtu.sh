##!/usr/bin/env bash


DTU_TESTING="/data1/local_userdata/wangshaoqian/dtu_testing/"


CKPT_FILE="./checkpoints/Effi_MVS_plus/model_dtu.ckpt"

OUT_DIR="./output/dtu/Effi_MVS_plus"

if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi

CUDA_VISIBLE_DEVICES=7 python test_dtu_dypcd.py --dataset=general_eval --batch_size=1 --testpath=$DTU_TESTING --GRUiters 3,3,3 --ndepths 48,8,8 --CostNum=3 --numdepth=384 --initloss=initloss --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --outdir $OUT_DIR --data_type dtu \
              --num_view=5 --dispmaxfirst=last --interval_scale=0.53
