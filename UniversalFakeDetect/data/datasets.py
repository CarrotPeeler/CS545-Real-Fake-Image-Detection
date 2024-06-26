import os
import pickle
import random as rand
from io import BytesIO
from random import choice, random, shuffle

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {"imagenet": [0.485, 0.456, 0.406], "clip": [0.48145466, 0.4578275, 0.40821073]}

STD = {"imagenet": [0.229, 0.224, 0.225], "clip": [0.26862954, 0.26130258, 0.27577711]}


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self, opt, al_mode=None, init_idxs=None):
        rand.seed(opt.seed)
        assert opt.data_label in ["train", "val"]
        # assert opt.data_mode in ["ours", "wang2020", "ours_wang2020"]
        self.opt = opt
        self.data_label = opt.data_label
        self.al_mode = al_mode
        self.init_idxs = init_idxs

        if opt.data_mode == "ours":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
        elif opt.data_mode == "wang2020":
            temp = "train/progan" if opt.data_label == "train" else "test/progan"
            real_list = get_list(os.path.join(opt.wang2020_data_path, temp), must_contain="0_real")
            fake_list = get_list(os.path.join(opt.wang2020_data_path, temp), must_contain="1_fake")
        elif opt.data_mode == "ours_wang2020":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
            temp = "train/progan" if opt.data_label == "train" else "test/progan"
            real_list += get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="0_real"
            )
            fake_list += get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="1_fake"
            )
        elif opt.data_mode == "dip":
            if opt.data_label == "train":
                path = "ImageData/train"
                real_subdirs = ["afhq-v2", "cc3m", "ffhq"]
                fake_subdirs = [
                    "IFv1-CC1M/IFv1-dpmsolver++-50-1M",
                    "SDv15R-CC1M/SDv15R-dpmsolver-25-1M",
                    "stylegan3-80K/stylegan3-r-afhqv2-512x512",
                    "stylegan3-80K/stylegan3-r-ffhqu-1024x1024",
                    "stylegan3-80K/stylegan3-r-metfaces-1024x1024",
                    "stylegan3-80K/stylegan3-t-afhqv2-512x512",
                    "stylegan3-80K/stylegan3-t-ffhqu-1024x1024",
                    "stylegan3-80K/stylegan3-t-metfaces-1024x1024",
                ]

            else:
                path = "ImageData/val"
                real_subdirs = ["celeba-hq", "cc3m"]
                fake_subdirs = [
                    "cogview2-22K",
                    "IF-CC95K/IF-ddim-50-15K",
                    "IF-CC95K/IF-ddpm-50-15K",
                    "IF-CC95K/IF-dpmsolver++-10-15K",
                    "IF-CC95K/IF-dpmsolver++-25-15K",
                    "Midjourneyv5-5K",
                    "SDv15-CC30K/SDv15R-dpmsolver-25-15K",
                    "SDv21-CC15K/SDv2-dpmsolver-25-10K",
                    "stylegan3-60K/stylegan3-r-afhqv2-512x512",
                    "stylegan3-60K/stylegan3-t-afhqv2-512x512",
                    "stylegan3-60K/stylegan3-t-ffhqu-1024x1024",
                    "stylegan3-60K/stylegan3-t-metfaces-1024x1024",
                ]

            real_list, fake_list = [], []

            for r_subdir in real_subdirs:
                r_temp = f"{path}/{r_subdir}"
                r_list = get_list(os.path.join(opt.wang2020_data_path, r_temp))
                if opt.max_sample is not None and opt.uniform_sample:
                    shuffle(r_list)
                    r_list = r_list[0 : opt.max_sample]
                real_list.extend(r_list)

            for f_subdir in fake_subdirs:
                f_temp = f"{path}/{f_subdir}"
                f_list = get_list(os.path.join(opt.wang2020_data_path, f_temp))
                if opt.max_sample is not None and opt.uniform_sample:
                    shuffle(f_list)
                    f_list = f_list[0 : opt.max_sample]
                fake_list.extend(f_list)

        if opt.max_sample is not None and not opt.uniform_sample:
            if (opt.max_sample > len(real_list)) or (opt.max_sample > len(fake_list)):
                opt.max_sample = 100
                print("not enough images, max_sample falling to 100")
            shuffle(real_list)
            shuffle(fake_list)
            real_list = real_list[0 : opt.max_sample]
            fake_list = fake_list[0 : opt.max_sample]

        real_list, fake_list = np.ma.array(real_list), np.ma.array(fake_list)

        if opt.use_active_learning and al_mode is not None:
            if al_mode == "init":
                real_list, fake_list = real_list[init_idxs], fake_list[init_idxs]
            elif al_mode == "pool":
                real_list[init_idxs] = np.ma.masked
                real_list = real_list[real_list.mask == False]

                fake_list[init_idxs] = np.ma.masked
                fake_list = fake_list[fake_list.mask == False]

        # setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0.0
        for i in fake_list:
            self.labels_dict[i] = 1.0

        self.total_list = np.concatenate([real_list.data, fake_list.data], axis=0)
        shuffle(self.total_list)
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = DoNothing()
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = DoNothing()
        if not opt.isTrain and opt.no_resize:
            rz_func = DoNothing()
        else:
            rz_func = CustomResize(opt)

        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"

        print("mean and std stats are from: ", stat_from)
        if "2b" not in opt.arch:
            print("using Official CLIP's normalization")
            self.transform = transforms.Compose(
                [
                    rz_func,
                    DataAugment(opt),
                    crop_func,
                    flip_func,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
                ]
            )
        else:
            print("Using CLIP 2B transform")
            self.transform = None  # will be initialized in trainer.py

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # shift all pool sample idxs by the size of the init dataset; prevents idx collision for dataset concatenation
        if self.opt.use_active_learning and self.al_mode == "pool":
            idx += len(self.init_idxs) * 2
        return img, label, idx


class DataAugment(torch.nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def forward(self, img):  # we assume inputs are always structured like this
        return data_augment(img, self.opt)


class CustomResize(torch.nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def forward(self, img):  # we assume inputs are always structured like this
        return custom_resize(img, self.opt)


class DoNothing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):  # we assume inputs are always structured like this
        return img


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
