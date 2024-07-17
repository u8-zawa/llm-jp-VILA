# This file is modified from https://github.com/NVlabs/VILA/tree/48aadd55c450b182f82f88ad340800428fa3a161

import argparse
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoConfig

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.model.multimodal_encoder.clyp.configuration_clyp import CLYPConfig


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


def eval_model(args):
    disable_torch_init()
    AutoConfig.register("clyp", CLYPConfig)


    # instruction tuned model
    # model_checkpoint_path = "checkpoints/clyp-llm-jp-v2_1_mm-align_en_3_sft_v1-5en_20240702"
    # model_checkpoint_path = "checkpoints/clyp-llm-jp-v2_1_mm-align_ja_3_sft_v1-5ja_20240703"
    model_checkpoint_path = args.model_path

    model_name = get_model_name_from_path(model_checkpoint_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_checkpoint_path, model_name)


    # image_files = [
    #     "https://raw.githubusercontent.com/tosiyuki/LLaVA-JP/main/imgs/sample1.jpg"
    #     # "https://raw.githubusercontent.com/tosiyuki/LLaVA-JP/main/imgs/sample2.jpg"
    # ]
    image_files = image_parser(args)
    images = load_images(image_files)


    # qs = "<image>\nこの画像には何が写っていますか。"
    # qs = "<image>\n Please describe this image."
    # qs = "<image>\n猫の隣には何がありますか？"
    # qs = "<image>\nこの画像の面白い点を教えてください。"
    qs = args.query

    conv_mode = args.conv_mode # "llmjp_v2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("prompt: ", prompt)


    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print("image tensor shape:", images_tensor.shape)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            # stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[: -len(stop_str)]
    # outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/clyp-llm-jp-v2_1_mm-align_en_3_sft_v1-5en_20240702")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, required=True, default="<image>\n Please describe this image.")
    parser.add_argument("--conv-mode", type=str, default="llmjp_v2")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    eval_model(args)
