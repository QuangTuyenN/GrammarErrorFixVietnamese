import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

model_path = "quangtuyennguyen/mT5-small-grammar-fix-vi"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

input_text = "hoom nay ddi bầu cử trongg nieemf vui haan hoan."

# Tokenize the input sentence
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=32, num_beams=4, early_stopping=True)

fix_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"Error Seq: {input_text}")
print(f"Fix: {fix_text}")
