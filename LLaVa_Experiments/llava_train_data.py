import json
import os
import random
import re
from io import BytesIO
from uuid import uuid4

import requests
import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from tqdm import tqdm

model_path = "liuhaotian/llava-v1.6-vicuna-7b"


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


# Model
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path, model_base=None, model_name=model_name
)


def eval_model(args):

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is "
            "{}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


unguided_prompt = (
    "Consider yourself to be an expert at differentiating real images "
    "from fake images generated with popular image generation techniques "
    "such as StyleGAN, Stable Diffusion, Midjourney, CogView, etc. "
    "The given image can be either real or fake. "
    "GAN, diffusion, etc. methods are quite powerful and can even fool humans, "
    "so be as analytical as possible. "
    "Please explain why you think the given image is real or fake. "
    "After the explanation, summarize the answer in one word in the format "
    "'Answer: real' or 'Answer: fake'"
)


parent_dir = "/hdd/shouborno/real-fake"

train_data = "ImageData/train"
real_subdirs = {"afhq-v2": "real", "cc3m": "real", "ffhq": "real"}
fake_subdirs = {
    "IFv1-CC1M/IFv1-dpmsolver++-50-1M": "DeepFloyd IF",
    "SDv15R-CC1M/SDv15R-dpmsolver-25-1M": "Stable Diffusion",
    "stylegan3-80K/stylegan3-r-afhqv2-512x512": "StyleGAN3",
    "stylegan3-80K/stylegan3-r-ffhqu-1024x1024": "StyleGAN3",
    "stylegan3-80K/stylegan3-r-metfaces-1024x1024": "StyleGAN3",
    "stylegan3-80K/stylegan3-t-afhqv2-512x512": "StyleGAN3",
    "stylegan3-80K/stylegan3-t-ffhqu-1024x1024": "StyleGAN3",
    "stylegan3-80K/stylegan3-t-metfaces-1024x1024": "StyleGAN3",
}


def generate_train_response(subdir, label, image_file):
    if label == "real":
        label_guide = "The given image is real."
    else:
        label_guide = f"The given image is a fake image generated by {fake_subdirs[subdir]}."

    guided_prompt = (
        "Consider yourself to be an expert at differentiating real images "
        "from fake images generated with popular image generation techniques "
        "such as StyleGAN, Stable Diffusion, Midjourney, CogView, etc. "
        f"{label_guide} "
        "GAN, diffusion, etc. methods are quite powerful and can even fool humans, "
        "so be as analytical as possible. "
        f"Please describe what makes it apparent that {label_guide} "
        "After the expanation, write the single-word summary of the answer "
        f"as 'Answer: {label}'"
    )

    args = type(
        "Args",
        (),
        {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": guided_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 200,
        },
    )()

    outputs = eval_model(args)
    return outputs


def generate_llava_finetuning_data(subdir_list, label, per_dataset_samples):
    train_json_list = []
    for subdir in tqdm(subdir_list):
        dirpath = os.path.join(parent_dir, train_data, subdir)
        image_filepaths = os.listdir(dirpath)
        samples = random.choices(image_filepaths, k=per_dataset_samples)
        for sample in tqdm(samples, leave=False):
            image_filepath = os.path.join(dirpath, sample)
            explanation = generate_train_response(subdir, label, image_filepath)
            train_json_list.append(
                {
                    "id": str(uuid4()),
                    "image": os.path.join(train_data, subdir, sample),
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{unguided_prompt}"},
                        {"from": "gpt", "value": explanation},
                    ],
                }
            )
    return train_json_list


real_json_list = generate_llava_finetuning_data(real_subdirs, "real", 500)
fake_json_list = generate_llava_finetuning_data(fake_subdirs, "fake", 500)
total_json_list = real_json_list + fake_json_list
random.shuffle(total_json_list)


with open("real_fake_llava_train.json", "w") as f:
    json.dump(total_json_list, f)
