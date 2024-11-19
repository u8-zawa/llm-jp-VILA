# Data Preparation

The following table shows an overview of the datasets used for training.

| Step | Language | Dataset | Images|
|:---|:---|:---|---:|
| Step-0 |Japanese|[Japanese image text pairs](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-image-text-pairs)|558K |
|        |English |[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)|558K |
| Step-1 |Japanese|[Japanese image text pairs](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-image-text-pairs)| 6M |
|        |        |[Japanese interleaved data](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-interleaved-data)| 6M |
|        |English |[coyo](https://github.com/kakaobrain/coyo-dataset) (subset) | 6M | 
|        |        |[mmc4-core](https://github.com/allenai/mmc4) (subset) | 6M | 
| Step-2 |Japanese|[llava-instruct-ja](https://huggingface.co/datasets/llm-jp/llava-instruct-ja)| 156K |
|        |        |[japanese-photos-conv](https://huggingface.co/datasets/llm-jp/japanese-photos-conversation)| 12K |
|        |        |[ja-vg-vqa](https://huggingface.co/datasets/llm-jp/ja-vg-vqa-conversation)| 99K |
|        |        |[synthdog-ja](https://huggingface.co/datasets/naver-clova-ix/synthdog-ja) (subset) | 102K |
|        |English |[LLaVA](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | 158K | 
|        |        |[VQAv2](https://visualqa.org/) | 53K | 
|        |        |[GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) | 46K | 
|        |        |[OCRVQA](https://ocr-vqa.github.io/) | 80K | 
|        |        |[TextVQA](https://textvqa.org/dataset/) | 22K | 

## Step-0

### Japanese image text pairs (Step-0)

1. Download [llm-jp_mm_pair_step0_558k.json](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-image-text-pairs/-/blob/main/llm-jp_mm_pair_step0_558k.json?ref_type=heads) and place it in the `playground/data/alt_pair_ja/` directory.

2. Download each image and save it in the `playground/data/alt_pair_ja/image_step0/` directory. Each image file should be named according to the value of the "image" key.

### LLaVA-Pretrain

We used [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) for training.

## Step-1

You may find it helpful to refer to the code from the original [VILA Repository](https://github.com/NVlabs/VILA/tree/48aadd55c450b182f82f88ad340800428fa3a161/data_prepare).

### Japanese image text pairs (Step-1)

1. Download all the jsonl files from [here](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-image-text-pairs).

2. Download each image, convert it to a Base64 string, and store it as the value of the `"image"` key.

    ```python
    img = Image.open(file_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
    ```

3. Save each jsonl file as a pickle object.

4. Count the number of data in each pickle file and save the number in `filename.count`. `filename.pkl` and `filename.count` should be in the `playground/data/alt_pair_ja/pkl02` directory.

### Japanese interleaved data

1. Download all the jsonl files from [here](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-japanese-interleaved-data).

2. Download each image, convert it to a Base64 string, and store it as the value of the `"image_base64"` key.

3. Save each jsonl file as a pickle object.

4. Count the number of data in each pickle file and save it in `filename.count`. `filename.pkl` and `filename.count` should be in the `playground/data/interleaved_ja/pkl-limit-tokens` directory.

### mmc4-core (subset)

In this [file](mdx/mmc4/mmc4_6m_ids.csv.gz), we provide the shard number and index within each shard for all samples in our training dataset.
Each line of this file contains `shard_number,index`.
This indicates that this sample is the `index`th sample in `docs_{shard_number}_v3.pkl`.
Note that we used non-fewer-face version.

### coyo (subset)

We share the ids of the data we use in this [file](mdx/coyo/coyo_6m_ids.txt.gz).
Each line of this file contains the data id.

## Step-2

### LLaVA-1.5 Instruction Data (subset)

We used a subset of the [LLaVA-1.5 Instruction Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) for training.

1. Download [llava_v1_5_subset_358k.json](https://huggingface.co/datasets/llm-jp/llava-instruct-v1_5-en-subset-358k/blob/main/llava_v1_5_subset_358k.json) and place it in the `playground/data/LLaVA-Instruct-150K/` directory.

2. Download the images from each dataset. You may find it helpful to refer to the [README in the original LLaVA repository](https://github.com/haotian-liu/LLaVA/blob/main/README.md#visual-instruction-tuning).


### llava-instruct-ja

1. Download [llava_instruct_ja_156k.json](https://huggingface.co/datasets/llm-jp/llava-instruct-ja/blob/main/llava_instruct_ja_156k.json) and place it in the `playground/data/llava_instruct_ja/` directory.

2. Download images for COCO dataset. Place [train2017](http://images.cocodataset.org/zips/train2017.zip) in the `playground/data/coco/train2017/` directory.

### japanese-photos-conv

1. Download [japanese_photos_conv_12k.json](https://huggingface.co/datasets/llm-jp/japanese-photos-conversation/blob/main/japanese_photos_conv_12k.json) and place it in the `playground/data/japanese-photos/` directory.

2. Download images from https://huggingface.co/datasets/ThePioneer/japanese-photos.

### ja-vg-vqa

1. Download [ja-vg-vqa_instruct_99k.json](https://huggingface.co/datasets/llm-jp/ja-vg-vqa-conversation/blob/main/ja-vg-vqa_instruct_99k.json) and place it in the `playground/data/ja-vg-vqa/` directory.

2. Download images for VisualGenome dataset.

### synthdog-ja (subset)

1. Download all parquet files from [here](https://huggingface.co/datasets/naver-clova-ix/synthdog-ja/tree/main/data) and place them in the `playground/data/synthdog-ja/data` directory.

2. Run [mdx/synthdog-ja/convert.py](mdx/synthdog-ja/convert.py)
