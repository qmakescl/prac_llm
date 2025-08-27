from dotenv import load_dotenv
import os

load_dotenv()

from huggingface_hub import whoami
# login(token=os.getenv("HUGGINGFACE_LOGIN_TOKEN"))
print( whoami() )

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = torch.bfloat16,
)

model_id = "google/gemma-2b-it"     # instruction-tuned model
# model_id = "google/gemma-2b"       # base model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, add_bos_token=True)

