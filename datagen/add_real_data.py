import argparse
import os
import shutil
from glob import glob

from img2dataset import download
from make_sentry_subset import batch_convert_png_jpg, generate_metadata


def download_cc3m(local_dataset_path):
    cc3m_train_anno = "datagen/cc3m/Train_GCC-training.tsv"
    cc3m_val_anno = "datagen/cc3m/Validation_GCC-1.1.0-Validation.tsv"
    cc3m_train_dir = f"{local_dataset_path}/ImageData/train/cc3m"
    cc3m_val_dir = f"{local_dataset_path}/ImageData/val/cc3m"

    for split in [(cc3m_train_anno, cc3m_train_dir), (cc3m_val_anno, cc3m_val_dir)]:
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
        # filter out non-jpg files and move jpg files to parent
        exts = [".parquet", ".txt", ".json", ".jpg"]
        for ext in exts:
            files = glob(f"{split[1]}/**/*{ext}", recursive=True)
            for file in files:
                if ext == ".jpg":
                    os.rename(file, f"{split[1]}/{file.rpartition('/')[-1]}")
                else:
                    os.remove(file)
        # remove empty folders after move
        subdirs = [f"{split[1]}/{f}" for f in list(os.listdir(split[1])) if "." not in f]
        for subdir in subdirs:
            shutil.rmtree(subdir)

    train_jpg_paths = glob(f"{cc3m_train_dir}/*.jpg")
    val_jpg_paths = glob(f"{cc3m_val_dir}/*.jpg")
    return train_jpg_paths, val_jpg_paths


def download_ffhq(local_dataset_path):
    dir_path = f"{local_dataset_path}/ImageData/train/ffhq"
    dir_path_tmp = f"{dir_path}-temp"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path_tmp, exist_ok=True)
    os.system(
        f"kaggle datasets download -d denislukovnikov/ffhq256-images-only -p {dir_path_tmp} \
              && unzip -j {dir_path_tmp}/ffhq256-images-only.zip -d {dir_path_tmp}"
    )
    os.remove(f"{dir_path_tmp}/ffhq256-images-only.zip")

    png_paths = glob(f"{dir_path_tmp}/*.png")
    jpg_paths = batch_convert_png_jpg(png_paths)
    shutil.rmtree(dir_path_tmp)
    return jpg_paths


def download_afhq_v2(local_dataset_path):
    dir_path = f"{local_dataset_path}/ImageData/train/afhq-v2"
    dir_path_tmp = f"{dir_path}-temp"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(dir_path_tmp, exist_ok=True)
    os.system(
        f"wget -N https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0 -O {dir_path_tmp}/afhq-v2.zip \
              && unzip -j {dir_path_tmp}/afhq-v2.zip -d {dir_path_tmp}"
    )
    os.remove(f"{dir_path_tmp}/afhq-v2.zip")

    png_paths = glob(f"{dir_path_tmp}/*.png")
    jpg_paths = batch_convert_png_jpg(png_paths)
    shutil.rmtree(dir_path_tmp)
    return jpg_paths


def download_celeba_hq(local_dataset_path):
    dir_path = f"{local_dataset_path}/ImageData/val/celeba-hq"
    os.makedirs(dir_path, exist_ok=True)
    os.system(
        f"kaggle datasets download -d badasstechie/celebahq-resized-256x256 -p {dir_path} \
              && unzip -j {dir_path}/celebahq-resized-256x256.zip -d {dir_path}"
    )
    os.remove(f"{dir_path}/celebahq-resized-256x256.zip")
    jpg_paths = glob(f"{dir_path}/*.jpg")
    return jpg_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--num_cores", type=int, default=10, help="number of cpu cores your system has"
    )
    parser.add_argument(
        "--total_threads", type=int, default=20, help="number of cpu threads total"
    )
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default="./sentry-dataset",
        help="path to where sentry-dataset is located",
    )
    args = parser.parse_args()

    jpg_paths_by_source = {}

    # download cc3m, ffhq, afhq-v2, and celeba_hq
    cc3m_train_jpgs, cc3m_val_jpgs = download_cc3m(args.local_dataset_path)
    jpg_paths_by_source["ImageData/train/cc3m"] = cc3m_train_jpgs
    jpg_paths_by_source["ImageData/val/cc3m"] = cc3m_val_jpgs
    jpg_paths_by_source["ImageData/train/ffhq"] = download_ffhq(args.local_dataset_path)
    jpg_paths_by_source["ImageData/train/afhq-v2"] = download_afhq_v2(args.local_dataset_path)
    jpg_paths_by_source["ImageData/val/celeba-hq"] = download_celeba_hq(args.local_dataset_path)

    # generate metadata
    generate_metadata(jpg_paths_by_source, args.local_dataset_path, label=1)
