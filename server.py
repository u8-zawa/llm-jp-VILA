import base64
import torch
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

app = Flask(__name__)

# グローバル変数としてモデルを保持
tokenizer = None
model = None
image_processor = None
context_len = None


def initialize_model():
    """モデルの初期化"""
    global tokenizer, model, image_processor, context_len
    
    print("モデルを初期化中...")
    disable_torch_init()
    
    model_checkpoint_path = "llm-jp/llm-jp-3-vila-14b"
    model_name = get_model_name_from_path(model_checkpoint_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_checkpoint_path, model_name
    )
    print("モデル初期化完了")


def load_image_from_url(url, timeout=10, max_size=10*1024*1024):
    """URLから画像をダウンロードして読み込み"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Content-Lengthをチェック（利用可能な場合）
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            raise ValueError(f"画像サイズが大きすぎます: {content_length} bytes")
        
        # 画像データを読み込み（最大サイズ制限付き）
        image_data = BytesIO()
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded > max_size:
                raise ValueError(f"画像サイズが大きすぎます: {downloaded} bytes")
            image_data.write(chunk)
        
        image_data.seek(0)
        image = Image.open(image_data).convert("RGB")
        return image
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"画像のダウンロードに失敗しました: {str(e)}")
    except Exception as e:
        raise ValueError(f"画像の処理に失敗しました: {str(e)}")


def decode_base64_image(base64_string):
    """Base64文字列から画像に変換"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Base64画像のデコードに失敗しました: {str(e)}")


def process_image_input(image_input):
    """画像入力を処理（Base64またはURL）"""
    if image_input.startswith('data:image'):
        # data:image/png;base64,... 形式
        try:
            base64_data = image_input.split(',')[1]
            return decode_base64_image(base64_data)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Data URL形式が不正です: {str(e)}")
    
    elif image_input.startswith('http://') or image_input.startswith('https://'):
        # HTTP(S) URL
        return load_image_from_url(image_input)
    
    else:
        # 直接のBase64文字列として扱う
        return decode_base64_image(image_input)


@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI ChatCompletions互換エンドポイント"""
    try:
        data = request.json
        
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.0)
        max_tokens = data.get('max_tokens', 256)
        
        if not messages:
            return jsonify({'error': 'messages are required'}), 400
        
        # 最後のメッセージから画像とテキストを抽出
        last_message = messages[-1]
        content = last_message.get('content', [])
        
        text_content = ""
        image_input = None
        
        for item in content:
            if item.get('type') == 'text':
                text_content = item.get('text', '')
            elif item.get('type') == 'image_url':
                image_url_data = item.get('image_url', {})
                image_input = image_url_data.get('url', '')
        
        if not text_content or not image_input:
            return jsonify({'error': 'Both text and image are required'}), 400
        
        # 画像を処理（URLまたはBase64）
        image = process_image_input(image_input)
        
        # プロンプトを構築
        query = f"<image>\n{text_content}"
        conv_mode = "llmjp_v3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()
        
        # 画像とテキストを処理
        images_tensor = process_images(
            [image], image_processor, model.config
        ).to(model.device, dtype=torch.float16)
        
        input_ids = tokenizer_image_token(
            formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        
        # テキスト生成
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                num_beams=1,
                max_new_tokens=max_tokens,
                use_cache=True,
            )
        
        # 結果をデコード
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # OpenAI互換レスポンス
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": outputs
                    },
                    "finish_reason": "stop"
                }
            ],
            "model": "llm-jp-3-vila-14b"
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=8000, debug=False)
