import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

@spaces.GPU
def translate(text):
    if not text.strip():
        return "請輸入要翻譯的中文"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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