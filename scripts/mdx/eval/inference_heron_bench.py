import argparse
import os
import json

import torch
from tqdm import tqdm
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


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        out.append(image)
    return out


def main(args):
    disable_torch_init()
    # AutoConfig.register("clyp", CLYPConfig)

    with open(args.question_file_path, "r") as f:
        questions = [json.loads(line) for line in f]

    model_checkpoint_path = args.model_path
    model_name = get_model_name_from_path(model_checkpoint_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_checkpoint_path, model_name)
    model_id = os.path.basename(model_checkpoint_path.rstrip("/"))

    results = []
    for question in tqdm(questions):
        qs = "<image>\n" + question["text"]

        conv_mode = args.conv_mode # "llmjp_v3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_file_path = os.path.join(args.image_dir_path, question["image"])
        images = load_images([image_file_path])
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[
                    images_tensor,
                ],
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        result = {
            "question_id": question["question_id"],
            "images": question["image"],
            "image_category": question["category"],
            "prompt": question["text"],
            "answer_id": "",
            "model_id": model_id,
            "metadata": {},
            "text": outputs,
        }
        results.append(result)

    base_name = f"run_{model_id}.jsonl"
    output_file_path = os.path.join(args.output_dir_path, base_name)
    if not os.path.isdir(args.output_dir_path):
        os.makedirs(args.output_dir_path)

    json_str = "\n".join([json.dumps(result, ensure_ascii=False, separators=(",", ":")) for result in results]) + "\n"
    with open(output_file_path, "w") as fout:
        fout.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/llm-jp-3-13b-instruct_siglip_mlp2xgelu_step-2_20241004")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llmjp_v3")
    parser.add_argument("--question_file_path", type=str)
    parser.add_argument("--image_dir_path", type=str)
    parser.add_argument("--output_dir_path", type=str, default=f"{os.path.dirname(__file__)}/output/")
    # parser.add_argument("--temperature", type=float, default=0)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)
