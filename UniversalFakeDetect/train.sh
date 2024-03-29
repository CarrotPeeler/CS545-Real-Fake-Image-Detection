python3 UniversalFakeDetect/train.py --name=clip_rn50_al --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip  --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=1 --gpu_ids=0 --num_threads=4 --batch_size=256 \
    --use_active_learning --niter=50 --dropout_iter=50 --query=1000 --acq_func=2 < /dev/null > clip_rn50_al.txt 2>&1 &
