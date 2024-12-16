from io import BytesIO

import requests
import torch
from PIL import Image
import gradio as gr

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


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


def main():
    disable_torch_init()

    model_checkpoint_path = "llm-jp/llm-jp-3-vila-14b"
    model_name = get_model_name_from_path(model_checkpoint_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_checkpoint_path, model_name)

    def chat_with_image(history, image_path, user_input):
        if image_path:
            image_files = [
                image_path
            ]
            images = load_images(image_files)
            image_tensors = [process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)]
        else:
            image_tensors = None

        conv_mode = "llmjp_v3"
        conv = conv_templates[conv_mode].copy()
        for i, (user, assistant) in enumerate(history + [(user_input, None)]):
            if image_path and i == 0:
                user = "<image>\n" + user
            conv.append_message(conv.roles[0], user)
            conv.append_message(conv.roles[1], assistant)
        
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=False,
                num_beams=1,
                max_new_tokens=256,
                use_cache=True,
            )

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        history.append([user_input, response])

        return history, history, ""

    with gr.Blocks() as demo:
        gr.Markdown("# LLM-jp-3 VILA ([llm-jp/llm-jp-3-vila-14b](https://huggingface.co/llm-jp/llm-jp-3-vila-14b))")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Image", type="filepath", optional=True)
                user_input = gr.Textbox(label="Input", placeholder="Type your message here", lines=2)
                send_button = gr.Button("Send")

            with gr.Column():
                chatbox = gr.Chatbot(label="Chat History")
                clear_button = gr.Button("Clear Chat")

        chat_history = gr.State(value=[])

        send_button.click(chat_with_image, [chat_history, image_input, user_input], [chatbox, chat_history, user_input])
        clear_button.click(lambda: ([], [], ""), None, [chatbox, chat_history, user_input])

    demo.launch()

if __name__ == "__main__":
    main()
