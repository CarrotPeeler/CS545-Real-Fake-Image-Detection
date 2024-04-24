python3 UniversalFakeDetect/train.py --name=TUNED2_clip_rn50_bal_weighted_loss_LWMS \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 --earlystop_epoch=5 \
    --use_active_learning --niter=5 --al_iter=50 --dropout_iter=10 --query=500 --num_samples_per_class=1000 --al_earlystop_metric=ap\
    --acq_func=9 --balance_acquisition --use_weighted_loss < /dev/null > train_bal_weighted_loss_LWMS_Tuned.txt 2>&1 &