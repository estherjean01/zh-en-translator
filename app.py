import gradio as gr
import spaces
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

@spaces.GPU
def translate(text):
    if not text.strip():
        return "請輸入要翻譯的中文"
    result = translator(text, max_length=512)
    return result[0]["translation_text"]

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=5, label="輸入中文", placeholder="在這裡輸入中文..."),
    outputs=gr.Textbox(label="英文翻譯"),
    title="中翻英小幫手",
    description="使用 Helsinki-NLP/opus-mt-zh-en 模型進行中文到英文翻譯",
    examples=[
        ["今天天氣很好，我們去公園散步吧。"],
        ["我正在學習如何使用 Hugging Face 建立應用程式。"],
    ],
)

demo.launch()