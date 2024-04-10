python3 UniversalFakeDetect/train.py --name=clip_rn50_al_bald_weighted \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 \
<<<<<<< HEAD
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=2 < /dev/null > train_al_max_entropy.txt 2>&1 &
=======
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=3 < /dev/null > train_al_bald_weighted.txt 2>&1 &
>>>>>>> 55c894d... Weighted loss implementation for Active Learning
