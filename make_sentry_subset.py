import os
import shutil
import argparse
from huggingface_hub import hf_hub_download, list_repo_tree



def recreate_file_struct_locally(dataset_path, image_dir_name, metadata_dir_name, repo_id):
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    os.makedirs(dataset_path, exist_ok=True)

    # recreate file structure locally
    for dir in [image_dir_name, metadata_dir_name]:
        for split in ['train', 'val']:
            subdirs = list(list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=f'{dir}/{split}'))
            subdirs = list(map(lambda x:x.path, subdirs))
            subdirs = list(filter(lambda x: x.rpartition('/')[-1][0] != '.', subdirs)) # filter out hidden files (i.e., .gitattributes)
            
            # recreate subdirs locally
            for subdir in subdirs:
                os.makedirs(f'{dataset_path}/{subdir}', exist_ok=True)

    return subdirs # list of all subdir paths for each data source





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

    subdirs = recreate_file_struct_locally(args.local_dataset_path, 
                                           args.image_dir, 
                                           args.metadata_dir, 
                                           args.repo_id)
    print(subdirs)
    for subdir in subdirs:
        print(list(list_repo_tree(repo_id="InfImagine/FakeImageDataset", repo_type="dataset", path_in_repo=subdir)))



    # hf_hub_download(repo_id="InfImagine/FakeImageDataset", filename="fleurs.py", repo_type="dataset")

    # p1 = list(list_repo_tree(repo_id="InfImagine/FakeImageDataset", repo_type="dataset", path_in_repo="ImageData"))
    # p1 = list(map(lambda x: x.path, p1))
    # print(p1)

   
        
    
    # fetch metadata
    #  = 
    # p1 = list(map(lambda x: x.path, p1))







