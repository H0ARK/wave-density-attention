import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)

messages = [
    {"role": "user", "content": "Explain what 2+2 is."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

print(f"--- PROMPT ---\n{prompt}\n--------------")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)

print(f"--- OUTPUT ---\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}\n--------------")
