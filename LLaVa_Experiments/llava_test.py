import os
import random
import re
from io import BytesIO

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

model_path = "checkpoints/llava-7b-real-fake-500/checkpoint-690"


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
    model_path=model_path, model_base=None, model_name=model_name, device="cuda:1"
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
        .to(model.device)
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


parent_dir = "/hdd/shouborno/real-fake"
val_path = "ImageData/val"
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


prompt = (
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


def infer_on_file(image_file):
    args = type(
        "Args",
        (),
        {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
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


def binary_text_classifier(text, image_file):
    binary_prompt = (
        "I'll end this prompt with an explanation regarding whether an image is real or fake. "
        "If the explanation is saying that the image is real, say real, "
        "and if it's saying that the image is fake, say fake. "
        "Clearly state your single-word answer as either real or fake. "
        f"Here's the explanation: {text}"
    )

    args = type(
        "Args",
        (),
        {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": binary_prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1,
        },
    )()

    outputs = eval_model(args)
    return outputs


def test_on_subdir(subdir_list, label, test_len=250):
    printable_results = ""
    for subdir in tqdm(subdir_list):
        dirpath = os.path.join(parent_dir, val_path, subdir)
        image_filepaths = os.listdir(dirpath)
        samples = random.choices(image_filepaths, k=test_len)

        correct = 0
        confused = 0
        for sample in tqdm(samples, leave=False):
            image_filepath = os.path.join(parent_dir, val_path, subdir, sample)
            printable_results += image_filepath + "\n"
            output = infer_on_file(image_filepath)
            printable_results += output + "\n"
            # output_words = output.split()
            # if not output_words[-2].lower()=="answer:":
            #     error_log = "ERROR: LLaVA ignored instruction!"
            #     print(error_log)
            #     printable_results+=error_log+"\n"
            # verdict = output_words[-1].lower()

            verdict = binary_text_classifier(output, image_filepath).lower()
            printable_results += "Verdict: " + verdict + "\n"
            if label in verdict or label[0] == verdict[0]:
                # Sometimes the LLM says "f" instead of "fake"
                correct += 1
            elif verdict != "real" and verdict != "fake":
                error_log = "ERROR: LLaVA ignored instruction!"
                confused += 1
                printable_results += error_log + "\n"

        accuracy = 100 * correct / test_len
        confusion_rate = 100 * confused / test_len
        print(subdir, "Accuracy", accuracy, "Confusion", confusion_rate)
    return printable_results


with open("test_explanations_and_classifications.txt", "w") as f:
    printable_results = test_on_subdir(real_subdirs, "real")
    f.write(printable_results)
    printable_results = test_on_subdir(fake_subdirs, "fake")
    f.write(printable_results)
