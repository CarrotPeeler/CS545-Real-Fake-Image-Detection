import os
import argparse
from img2dataset import download


def download_cc3m():
    cc3m_train_anno = "datagen/cc3m/Train_GCC-training.tsv"
    cc3m_val_anno = "datagen/cc3m/Validation_GCC-1.1.0-Validation.tsv"
    cc3m_train_dir = "./sentry-dataset/ImageData/train/CC3M"
    cc3m_val_dir = "./sentry-dataset/ImageData/val/CC3M"

    for split in [(cc3m_train_anno, cc3m_train_dir), 
                  (cc3m_val_anno, cc3m_val_dir)]:
        os.makedirs(split[1], exist_ok=True) 
        download(
            processes_count=args.num_cores,
            thread_count=args.total_threads,
            url_list=split[0],
            image_size=256,
            output_folder=split[1],
            output_format="files",
            encode_format="jpg",
            input_format="tsv",
            url_col="url",
            caption_col="caption",
            enable_wandb=False,
            number_sample_per_shard=1000,
            distributor="multiprocessing",
        )


def download_ffhq():
    pass


def download_afhq_v2():
    pass


def download_celeba_hq():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_cores", type=int, default=10, 
                        help="number of cpu cores your system has")
    parser.add_argument("--total_threads", type=int, default=20, 
                        help="number of cpu threads total")
    args = parser.parse_args()

    download_cc3m()
    download_ffhq()
    download_afhq_v2
    download_celeba_hq()


    