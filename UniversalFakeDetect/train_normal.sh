python3 UniversalFakeDetect/train.py \
    --name=clip_vitl14_pretrained \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip  --arch=CLIP:ViT-L/14 --fix_backbone --save_epoch_freq=1 \
    --gpu_ids=0 --num_threads=4 --batch_size=256 --niter=10 \
    --ckpt=./UniversalFakeDetect/pretrained_weights/fc_weights.pth < /dev/null > train_normal.txt 2>&1 &
