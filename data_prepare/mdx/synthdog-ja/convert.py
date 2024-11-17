import argparse
import os
import json
from io import BytesIO

import pandas as pd
from tqdm import tqdm
from PIL import Image


TEMPLATE_JA = (
    (
        "写真に写っている文字をすべて書き出してください。",
        "この写真に写っている文字を以下に示します。\n```\n",
        "\n```\n"
    ),
    (
        "画像の中の文字をすべて出力してください。",
        "入力された画像内の文字は以下の通りです。\n```\n", 
        "\n```\n"
    ),
    (
        "この画像には何と書かれていますか？",
        "この画像に書かれているテキストを以下に表示します。\n```\n",
        "\n```\n"
    ),
    (
        "画像に書かれているテキストを全て出力しなさい。",
        "この画像に書かれているテキストを以下に示します。\n```\n",
        "\n```\n"
    ),
    (
        "与えられた画像中のテキストを全て出力しなさい。",
        "画像中のテキストは以下の通りです。\n```\n",
        "\n```\n"
    ),
    (
        "この写真内の全てのテキストを抜き出してください。",
        "この写真内のテキストを以下に表示します。\n```\n",
        "\n```\n"
    ),
)


TEMPLATE_SIZE = len(TEMPLATE_JA)
OCRVAQ_SIZE = 80000
TEXTVQA_SIZE = 21953


def main(save_image: bool = False):
    dir_path = "./playground/data/synthdog-ja/data"
    file_list = [os.path.join(dir_path, file) for file in sorted(os.listdir(dir_path))
                 if file.endswith(".parquet")]

    output_list = []
    sum_images = 0
    template_idx = 0
    progress = tqdm(total=len(file_list), desc='Progress')
    for idx, file in enumerate(file_list):
        df = pd.read_parquet(file)
        data_list = df.to_dict("records")
        os.makedirs(f"./playground/data/synthdog-ja/image/{idx:03d}", exist_ok=True)

        for image_idx, data in enumerate(data_list):
            if sum_images == OCRVAQ_SIZE + TEXTVQA_SIZE:
                break

            ground_truth = json.loads(data["ground_truth"])["gt_parse"]["text_sequence"].replace(" ", "\n")

            if save_image:
                image = Image.open(BytesIO(data["image"]["bytes"]))
                image.save(
                    f"./playground/data/synthdog-ja/image/{idx:03d}/{image_idx:05d}.jpg",
                    format="JPEG"
                )

            conversation_list = [
                {
                    "from": "human",
                    "value": TEMPLATE_JA[template_idx][0]
                },
                {
                    "from": "gpt",
                    "value": TEMPLATE_JA[template_idx][1] + ground_truth + TEMPLATE_JA[template_idx][2]
                },
            ]

            data_dict = {
                "id": f"synthdog-ja/image/{idx:03d}/{image_idx:05d}",
                "image": f"synthdog-ja/image/{idx:03d}/{image_idx:05d}.jpg",
                "conversations": conversation_list
            }

            output_list.append(data_dict)
            sum_images += 1
            template_idx = (template_idx + 1) % TEMPLATE_SIZE

        if sum_images == OCRVAQ_SIZE + TEXTVQA_SIZE:
            break

        progress.update(1)

    progress.close()

    # Add <image>
    flag = True
    for i in range(len(output_list)):
        if flag:
            output_list[i]["conversations"][0]["value"] = "<image>\n" + output_list[i]["conversations"][0]["value"]
            flag = False
        else:
            output_list[i]["conversations"][0]["value"] = output_list[i]["conversations"][0]["value"] + "\n<image>"
            flag = True

    print(len(output_list))
    with open(f"./playground/data/synthdog-ja/synthdog_ja_{round(len(output_list)/1000)}k.json", "w") as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_image", type=bool, default=False)
    args = parser.parse_args()
    main(args.save_image)
