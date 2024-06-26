python3 UniversalFakeDetect/train.py --name=clip_rn50_al_max_entropy \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 \
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=2 < /dev/null > train_al_max_entropy.txt 2>&1 &
wait; python3 UniversalFakeDetect/train.py --name=clip_rn50_al_bald \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 \
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=3 < /dev/null > train_al_bald.txt 2>&1 &
wait; python3 UniversalFakeDetect/train.py --name=clip_rn50_al_var_ratios \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 \
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=4 < /dev/null > train_al_var_ratios.txt 2>&1 &
wait; python3 UniversalFakeDetect/train.py --name=clip_rn50_al_mean_std \
    --wang2020_data_path=/home/vislab-001/Jared/dip/CS545-Real-Fake-Image-Detection/sentry-dataset/ \
    --data_mode=dip --arch=CLIP:RN50 --fix_backbone --save_epoch_freq=5 --gpu_ids=0 --num_threads=4 --batch_size=256 \
    --use_active_learning --niter=30 --al_iter=20 --dropout_iter=10 --query=2500 --acq_func=5 < /dev/null > train_al_mean_std.txt 2>&1 &
