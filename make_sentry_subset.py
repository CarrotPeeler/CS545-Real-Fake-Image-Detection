import os
import shutil
import argparse
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
        if files are for train data, reduce sample size by train_reduce_factor
        if files are for val data, reduce sample size by val_reduce_factor
    args:
        files: list of tar files for one image data source
    """
    files_subset = []
    if 'MetaData' in files[0] or len(files) == 1: # if metadata or only 1 file, return as is
        return files
    elif 'ImageData/train' in files[0]:
        files_subset = files[0:int(len(files)/train_reduce_factor)]
    elif 'ImageData/val' in files[0]:
        files_subset = files[0:int(len(files)/val_reduce_factor)]
    return files_subset


def recreate_file_struct_locally(dataset_path, image_dir_name, metadata_dir_name, repo_id):
    """Locally reconstructs the folder structure for a dataset repo from HuggingFace Data Servers"""
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    os.makedirs(dataset_path, exist_ok=True)

    all_subdirs = []

    # recreate file structure locally
    for dir in [image_dir_name, metadata_dir_name]:
        for split in ['train', 'val']:
            if dir == image_dir_name:
                subdirs = list(list_repo_tree(repo_id=repo_id, 
                                              repo_type="dataset", 
                                              path_in_repo=f'{dir}/{split}',
                                              recursive=True))
                subdirs = list(map(lambda x:x.path, subdirs))
                subdirs = list(filter(lambda x: '.' not in x, subdirs)) # remove any files found
                subdirs = filter_redundant_paths(subdirs)
            else:
                subdirs = [f'{metadata_dir_name}/{split}']

            # recreate subdirs locally
            for subdir in subdirs:
                os.makedirs(f'{dataset_path}/{subdir}', exist_ok=True)

            all_subdirs.append(subdirs)

    return all_subdirs # list of all subdir paths for each data source


def download_files(all_subdirs,
                   repo_id, 
                   local_dataset_dir, 
                   train_reduce_factor=4, 
                   val_reduce_factor=2):
    for subdirs in all_subdirs:
        for subdir in subdirs:
            files = list(list_repo_tree(repo_id=repo_id, 
                                        repo_type="dataset", 
                                        path_in_repo=subdir,
                                        recursive=True))
            files = list(map(lambda x:x.path, files))
            files = filter_hidden_files(files)

            # develop subset of files 
            files = filter_imagedata_subset(files, train_reduce_factor, val_reduce_factor)

            print(files)
            # for file in files:
            #     local_save_dir = f"{local_dataset_dir}/{file.rpartition('/')[0]}"
            #     # download tar files from HuggingFace
            #     hf_hub_download(repo_id=repo_id, 
            #                     repo_type="dataset",
            #                     filename=file,
            #                     local_dir=local_save_dir)
            #     os.system(f"tar -xvf {local_save_dir}/{file.rpartition('/')[-1]}")

                # batch convert from png to jpg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--local_dataset_path', type=str, default='./sentry-dataset', 
                        help='path to where dataset will be locally saved')
    parser.add_argument('--image_dir', type=str, default='ImageData', 
                        help='name of dir where train and val dirs for image data located')
    parser.add_argument('--metadata_dir', type=str, default='MetaData', 
                        help='name of dir where train and val dirs for metadata located')
    parser.add_argument('--repo_id', type=str, default='InfImagine/FakeImageDataset', 
                        help='dataset repo id listed on huggingface')
       
    args = parser.parse_args()

    all_subdirs = recreate_file_struct_locally(args.local_dataset_path, 
                                           args.image_dir, 
                                           args.metadata_dir, 
                                           args.repo_id)
    # print(all_subdirs)
    download_files(all_subdirs, args.repo_id, args.local_dataset_path)


   
   
        
    
    # fetch metadata
    #  = 
    # p1 = list(map(lambda x: x.path, p1))







