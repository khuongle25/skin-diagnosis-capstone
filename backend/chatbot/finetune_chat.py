from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# login('')

# !pip install -U transformers datasets accelerate bitsandbytes flash-attn scikit-learn peft trl protobuf huggingface_hub sentencepiece

# --- Cell 2: Import thư viện ---
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import transformers
logger.info(f"Transformers version: {transformers.__version__}")
logger.info(f"CUDA is available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# --- Cell: Load model và trả lời câu hỏi (cho P100) ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from IPython.display import Markdown, display

# 1. Đường dẫn đến các model và checkpoint
model_id = "/kaggle/input/dermbot/pytorch/default/1/llama2-7b-lora-dermnetz"  # Base model
adapter_path = "/kaggle/input/dermbot/pytorch/default/1/llama2-7b-lora-dermnetz"  # Đường dẫn đến adapter

logger.info(f"Loading tokenizer from {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

logger.info(f"Loading base model from {model_id}...")
# Cấu hình cho P100 - KHÔNG sử dụng flash attention
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # P100 không hỗ trợ bfloat16, dùng float16
    device_map="auto",
    # BỎ use_flash_attention_2 - P100 không hỗ trợ
)

logger.info(f"Loading adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
# model.eval()  # Đặt model ở chế độ eval

logger.info("Model loaded successfully!")

# 2. Hàm generate câu trả lời
def generate_response(question, max_new_tokens=512, temperature=0.7):
    # Tạo prompt theo định dạng Llama-2-chat
    prompt = f"<s>[INST] {question} [/INST]"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    
    # Decode và trích xuất phần trả lời
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("[/INST]")[-1].strip()
    return answer
