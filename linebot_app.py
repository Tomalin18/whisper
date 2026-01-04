"""
Whisper LINE Bot - 會議記錄助手
1. 語音/音檔轉錄
2. GPT-4 整理成會議記錄
"""

import os
import tempfile
import threading
import requests
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage
)
from linebot.v3.webhooks import MessageEvent, AudioMessageContent, FileMessageContent
import whisper
from openai import OpenAI

# 支援的音訊格式
AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.flac', '.ogg', '.webm', '.mp4'}

# 載入 .env 檔案
load_dotenv()

app = Flask(__name__)

# ============================================
# API 設定 (從 .env 讀取)
# ============================================
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not CHANNEL_ACCESS_TOKEN or not CHANNEL_SECRET:
    raise ValueError("請設定 LINE_CHANNEL_ACCESS_TOKEN 和 LINE_CHANNEL_SECRET 環境變數")

if not OPENAI_API_KEY:
    raise ValueError("請設定 OPENAI_API_KEY 環境變數")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 載入 Whisper 模型
print("正在載入 Whisper 模型...")
model = whisper.load_model("medium")
print("模型載入完成！")

# 會議記錄整理 Prompt
MEETING_SUMMARY_PROMPT = """你是一個專業的會議記錄整理助手。請將以下逐字稿整理成結構化的會議記錄。

重要：
- 必須使用繁體中文（臺灣用語）
- 如果逐字稿包含簡體中文，請全部轉換為繁體中文

格式要求：
1. 會議摘要（2-3句話概述）
2. 討論重點（條列式）
3. 決議事項（如有）
4. 待辦事項（如有）

保持專業但易讀的風格。

逐字稿：
{transcript}
"""


def transcribe_audio(message_id, file_ext=".m4a"):
    """下載並轉錄音訊"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"下載失敗: {response.status_code}")

    # 暫存音訊並轉錄
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
        f.write(response.content)
        temp_path = f.name

    print(f"轉錄中: {temp_path}")
    result = model.transcribe(temp_path, language="zh")
    text = result.get("text", "").strip()

    # 清理暫存檔
    os.unlink(temp_path)

    return text if text else "（無法辨識語音內容）"


def summarize_meeting(transcript):
    """使用 GPT-4 整理會議記錄"""
    print("正在使用 GPT-4 整理會議記錄...")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一個專業的會議記錄整理助手。"},
            {"role": "user", "content": MEETING_SUMMARY_PROMPT.format(transcript=transcript)}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    
    return response.choices[0].message.content


def transcribe_and_summarize(user_id, message_id, file_ext=".m4a", filename="音訊"):
    """背景轉錄並整理會議記錄"""
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            
            # Step 1: 轉錄
            transcript = transcribe_audio(message_id, file_ext)
            print(f"轉錄完成: {transcript[:100]}...")
            
            # 先發送逐字稿
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=f"📝 逐字稿：\n\n{transcript[:1500]}{'...(內容過長已截斷)' if len(transcript) > 1500 else ''}")]
                )
            )
            
            # Step 2: GPT-4 整理
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text="🤖 正在使用 AI 整理會議記錄...")]
                )
            )
            
            summary = summarize_meeting(transcript)
            
            # 發送會議記錄
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=f"📋 會議記錄：\n\n{summary}")]
                )
            )
            
            print(f"會議記錄整理完成: {filename}")

    except Exception as e:
        print(f"處理錯誤: {e}")
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.push_message(
                PushMessageRequest(
                    to=user_id,
                    messages=[TextMessage(text=f"❌ 處理失敗: {str(e)}")]
                )
            )


@app.route("/callback", methods=["POST"])
def callback():
    """LINE Webhook 入口"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"Webhook 處理錯誤: {e}")
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio(event):
    """處理語音訊息"""
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="🎙️ 收到語音訊息，開始處理...\n\n1. 轉錄中\n2. 整理會議記錄")]
            )
        )

        thread = threading.Thread(
            target=transcribe_and_summarize,
            args=(event.source.user_id, event.message.id)
        )
        thread.start()


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event):
    """處理檔案上傳"""
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)

        filename = event.message.file_name
        ext = os.path.splitext(filename)[1].lower()

        if ext not in AUDIO_EXTENSIONS:
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=f"不支援的檔案格式: {ext}\n支援格式: {', '.join(AUDIO_EXTENSIONS)}")]
                )
            )
            return

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=f"📁 收到 {filename}，開始處理...\n\n1. 轉錄中\n2. 整理會議記錄")]
            )
        )

        thread = threading.Thread(
            target=transcribe_and_summarize,
            args=(event.source.user_id, event.message.id, ext, filename)
        )
        thread.start()


if __name__ == "__main__":
    print("啟動會議記錄助手...")
    print("Webhook URL: http://localhost:5001/callback")
    app.run(host="0.0.0.0", port=5001, debug=False)
