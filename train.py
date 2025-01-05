from huggingface_hub import login
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MT5ForConditionalGeneration


os.environ["HUGGINGFACE_WRITE_TOKEN"] = ''

login('HUGGINGFACE_WRITE_TOKEN')

dataset = load_dataset("bmd1905/error-correction-vi")

print(dataset)

train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# train_dataset.to_csv("train_data.csv", index=False)
# test_dataset.to_csv("test_data.csv", index=False)

model_id = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def preprocess_function(examples):
    inputs = [ex for ex in examples['error_text']]
    targets = [ex for ex in examples['correct_text']]

    model_inputs = tokenizer(inputs, max_length=32, truncation=True, padding=True, return_tensors='pt')

    # Tokenize the target (output) text
    labels = tokenizer(targets, max_length=32, truncation=True, padding=True, return_tensors='pt').input_ids

    # Replace padding token IDs in the labels with -100 to ignore them in loss calculation
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs


data_train_pt = train_dataset.map(preprocess_function, batched=True, batch_size=None)
data_test_pt = test_dataset.map(preprocess_function, batched=True, batch_size=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MT5ForConditionalGeneration.from_pretrained(model_id).to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="mT5-small-grammar-fix-vi",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)


# check all params are continuously
for name, param in model.named_parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=data_train_pt,
    eval_dataset=data_test_pt,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.push_to_hub()



