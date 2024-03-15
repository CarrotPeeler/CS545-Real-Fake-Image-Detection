import os
import shutil
import argparse
from glob import glob
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download, list_repo_tree


def filter_hidden_files(list_of_files):
    # filter out hidden files (i.e., .gitattributes)
    filtered = list(filter(lambda x: x.rpartition('/')[-1][0] != '.', list_of_files))
    return filtered


def filter_redundant_paths(subdirs):
    # remove redundant/overlapping paths (i.e., home/user/cd is redundant with home/user/cd/disco)
    to_remove = []
    for path1 in subdirs:
        for path2 in subdirs:
            # check if leaf node from path1 is in path2
            if path1 != path2 and path1.rpartition('/')[-1] in path2:
                to_remove.append(path1)

    filtered_subdirs = [path for path in subdirs if path not in to_remove]
    return filtered_subdirs


def filter_imagedata_subset(files, train_reduce_factor, val_reduce_factor):
    """
    Creates a subset of files
        if files are for train data, reduce sample size to train_reduce_factor
        if files are for val data, reduce sample size to val_reduce_factor
    args:
        files: list of tar files for one image data source
    """
    files_subset = []
    train_end_idx = max(int(train_reduce_factor*len(files)), 1)
    val_end_idx = max(int(val_reduce_factor*len(files)), 1)

    if "MetaData" in files[0] or len(files) == 1: # if metadata or only 1 file, return as is
        return files
    elif "ImageData/train" in files[0]:
        files_subset = files[0:train_end_idx]
    elif "ImageData/val" in files[0]:
        files_subset = files[0:val_end_idx]
    return files_subset


def recreate_file_struct_locally(dataset_path, image_dir_name, metadata_dir_name, repo_id):
    """Locally reconstructs the folder structure for a dataset repo from HuggingFace Data Servers"""
    os.makedirs(dataset_path, exist_ok=True)

    all_image_subdirs = []
    all_metadata_subdirs = []

    # recreate file structure locally
    for dir in [image_dir_name, metadata_dir_name]:
        for split in ["train", "val"]:
            if dir == image_dir_name:
                subdirs = list(list_repo_tree(repo_id=repo_id, 
                                              repo_type="dataset", 
                                              path_in_repo=f'{dir}/{split}',
                                              recursive=True))
                subdirs = list(map(lambda x:x.path, subdirs))
                subdirs = list(filter(lambda x: '.' not in x, subdirs)) # remove any files found
                subdirs = filter_redundant_paths(subdirs)
                all_image_subdirs.append(subdirs)
            else:
                subdirs = [f'{metadata_dir_name}/{split}']
                all_metadata_subdirs.append(subdirs)

            # recreate subdirs locally
            for subdir in subdirs:
                os.makedirs(f'{dataset_path}/{subdir}', exist_ok=True)

    return all_image_subdirs, all_metadata_subdirs # list of all subdir paths and csvs for each data source


def batch_convert_png_jpg(png_paths):
    """
    Batch convert png files to jpg format

    args:
        png_paths: list of string paths to png files
    returns:
        jpg_paths: list of string paths where jpgs saved
    """
    temp_dir = png_paths[0].rpartition('/')[0]
    jpg_paths = list(map(lambda x: x.replace("-temp", ""), png_paths))
    jpg_paths = list(map(lambda x: f"{x.rpartition('.')[0]}.jpg", jpg_paths))
    with open(temp_dir + "/ffmpeg_commands.sh", 'a+') as f:
        for i in range(len(png_paths)):
            f.writelines(f"ffmpeg -loglevel quiet -i {png_paths[i]} {jpg_paths[i]}\n")
    
    # parallelize ffmpeg commands
    os.system(f"parallel --eta < {temp_dir}/ffmpeg_commands.sh")
    return jpg_paths


def download_image_files_as_jpg(all_subdirs,
                                repo_id, 
                                local_dataset_dir, 
                                train_reduce_factor=.125, 
                                val_reduce_factor=.5):
    jpg_paths_by_source = {}

    for i in tqdm(range(len(all_subdirs))):
        for subdir in all_subdirs[i]:
            dataset_name = subdir.rpartition('/')[-1]
            os.system(f"echo DATASET: {dataset_name}")
            files = list(list_repo_tree(repo_id=repo_id, 
                                        repo_type="dataset", 
                                        path_in_repo=subdir,
                                        recursive=True))
            files = list(map(lambda x:x.path, files))
            files = filter_hidden_files(files)
            filesNumbered = True if len(files) > 1 else False

            # develop subset of files 
            files = filter_imagedata_subset(files, train_reduce_factor, val_reduce_factor)

            save_dir = f"{local_dataset_dir}/{subdir}" # dir where jpg images stored
            temp_save_dir = f"{save_dir}-temp" # temp dir where png images stored
            os.makedirs(temp_save_dir, exist_ok=True)

            # download all image files for a single data source
            for file in files:
                os.system(f"echo DOWNLOADING: {file.rpartition('/')[-1]}")
                file_path = f"{local_dataset_dir}/{file}"

                # download tar files from HuggingFace
                hf_hub_download(repo_id=repo_id, 
                                repo_type="dataset",
                                filename=file,
                                local_dir=local_dataset_dir, 
                                local_dir_use_symlinks=False)
                
            fname = f"{local_dataset_dir}/{file.partition('.')[0]}"
            if filesNumbered:
                # cat all tar files together
                os.system(f"cat {fname}.tar.gz.* > {fname}.tar.gz")
                
                # remove original tar files to reduce space
                for file in files:
                    file_path = f"{local_dataset_dir}/{file}"
                    os.remove(file_path)

            # extract final tar
            os.system(f"tar -xf {fname}.tar.gz --strip-components 1 -C {temp_save_dir}")
            # delete tar after
            os.remove(f"{fname}.tar.gz")

            # locate all png files from extracted subdirs
            png_paths = glob(f"{temp_save_dir}/**/*.png", recursive=True)

            # recreate subdirs in save_dir if necessary
            extracted_subdirs = list(set(map(lambda x: x.rpartition('/')[0], png_paths)))
            extracted_subdirs = list(map(lambda x: x.replace("-temp", ""), extracted_subdirs))
            for ext_dir in extracted_subdirs:
                os.makedirs(ext_dir, exist_ok=True)

            # batch convert from png to jpg
            jpg_paths = batch_convert_png_jpg(png_paths)
            jpg_paths_by_source[subdir] = jpg_paths

            # remove temp dir of png imgs after finishing conversion
            shutil.rmtree(temp_save_dir) 
    return jpg_paths_by_source


def generate_metadata(jpg_paths_by_source:dict, local_dataset_path):
    """Generate metadata annotations for image data"""
    for k,v in jpg_paths_by_source.items():
        # add target label 0 (fake) and newline char
        jpg_paths = list(map(lambda x: f"{x} 0\n", v))

        dataset_name = k.rpartition('/')[-1]
        metadata_path = k.replace("ImageData", "MetaData").rpartition('/')[0]

        with open(f"{local_dataset_path}/{metadata_path}/{dataset_name}.csv", "a+") as f:
            f.writelines(jpg_paths)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_dataset_path", type=str, default="./sentry-dataset", 
                        help="path to where dataset will be locally saved")
    parser.add_argument("--image_dir", type=str, default="ImageData", 
                        help="name of dir where train and val dirs for image data located")
    parser.add_argument("--metadata_dir", type=str, default="MetaData", 
                        help="name of dir where train and val dirs for metadata located")
    parser.add_argument("--repo_id", type=str, default="InfImagine/FakeImageDataset", 
                        help="dataset repo id listed on huggingface")
    args = parser.parse_args()

    # construct local folder system matching data repo on HuggingFace servers
    all_image_subdirs, all_metadata_subdirs = recreate_file_struct_locally(args.local_dataset_path, 
                                                                           args.image_dir, 
                                                                           args.metadata_dir, 
                                                                           args.repo_id)
    print(all_image_subdirs)
    # download all images in jpg format 
    jpg_paths_by_source = download_image_files_as_jpg([all_image_subdirs[1][1:]], 
                                                      args.repo_id, 
                                                      args.local_dataset_path)
    # generate all image annotations
    generate_metadata(jpg_paths_by_source, args.local_dataset_path)









