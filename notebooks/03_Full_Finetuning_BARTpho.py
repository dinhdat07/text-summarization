# %% [markdown]
# # Notebook 3: Full Fine-tuning BARTpho
# !pip install transformers datasets evaluate rouge_score accelerate
# YÊU CẦU: BẬT GPU T4 x1 TRÊN KAGGLE (Runtime -> Accelerator -> GPU T4)

# %%
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import torch

# %%
# 1. Tải dữ liệu
# KAGGLE TIP: Add data tạo từ Notebook 1 vào đây.
print("Loading datasets...")
train_df = pd.read_csv("train_10k.csv")
val_df = pd.read_csv("val_1k.csv")

# Bỏ qua các dòng NaN nếu có
train_df = train_df.dropna(subset=["article", "abstract"])
val_df = val_df.dropna(subset=["article", "abstract"])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# %%
# 2. Setup Tokenizer & Model
# Sử dụng bản syllable để không bị vướng dependency VnCoreNLP trên Kaggle
MODEL_NAME = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

    labels = tokenizer(text_target=examples["abstract"], max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Map data
print("Tokenizing data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_val = val_dataset.map(preprocess_function, batched=True, num_proc=4)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

# %%
# 3. Setup Trainer
# VRAM T4 16GB rất giới hạn cho Full FT seq2seq. Ta phải dùng fp16, batch_size nhỏ, grad_acc
training_args = Seq2SeqTrainingArguments(
    output_dir="./bartpho_full_ft_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch = 4 * 4 = 16
    weight_decay=0.01,
    save_total_limit=2, # Tránh đầy ổ cứng Kaggle
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True, # CRITICAL: Phải bật fp16
    logging_steps=100,
    report_to="none" # Bỏ report wandb để tránh rắc rối đăng nhập
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
# 4. Bắt đầu Train
print("Bắt đầu Full Fine-tuning (Có thể mất 4-6 tiếng)...")
trainer.train()

# Lưu model cuối cùng ra Output
trainer.save_model("./bartpho_full_ft_final")
tokenizer.save_pretrained("./bartpho_full_ft_final")
print("Đã lưu mô hình Full FT tại ./bartpho_full_ft_final")
# KAGGLE TIP: Hãy lưu output này thành một Dataset Kaggle để lấy ra inference ở Notebook 5
