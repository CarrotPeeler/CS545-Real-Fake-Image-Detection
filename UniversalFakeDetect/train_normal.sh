python3 UniversalFakeDetect/train.py --name=clip_rn50 --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip  --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 \
    --gpu_ids=0 --num_threads=4 --batch_size=256 --niter=40 < /dev/null > clip_rn50.txt 2>&1 &