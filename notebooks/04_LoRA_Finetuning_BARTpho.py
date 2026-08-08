# %% [markdown]
# # Notebook 4: LoRA Fine-tuning BARTpho
#
# YÊU CẦU: BẬT GPU T4 x2 TRÊN KAGGLE

# %%
# !pip install transformers datasets evaluate rouge_score accelerate peft "torchao>=0.16.0"

# %%
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

# %%
# 1. Tải dữ liệu
print("Loading datasets...")
train_df = pd.read_csv("train_10k.csv")
val_df = pd.read_csv("val_1k.csv")

train_df = train_df.dropna(subset=["article", "abstract"])
val_df = val_df.dropna(subset=["article", "abstract"])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# %%
# 2. Setup Tokenizer, Model & LoRA Config
MODEL_NAME = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# Cấu hình LoRA (Chỉ can thiệp khoảng ~1-2% tham số)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=16,          # Rank của matrix
    lora_alpha=32, # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"] # Gắn LoRA vào Attention block
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
# 3. Tiền xử lý
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(text_target=examples["abstract"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_val = val_dataset.map(preprocess_function, batched=True, num_proc=4)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
# 4. Setup Trainer
# VRAM của LoRA thấp hơn nhiều so với Full FT, ta có thể tăng batch size
training_args = Seq2SeqTrainingArguments(
    output_dir="./bartpho_lora_checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4, # LoRA thường cần learning rate cao hơn (e.g., 5e-4 thay vì 2e-5)
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2, # Effective batch = 8 * 2 = 16
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    logging_steps=100,
    report_to="none"
)

# Metric
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}

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
# 5. Bắt đầu Train
print("Bắt đầu LoRA Fine-tuning (Khoảng 2-3 tiếng)...")
trainer.train()

# Lưu model LoRA (Chỉ lưu phần adapter weights, file rất nhẹ ~vài MB)
model.save_pretrained("./bartpho_lora_final")
tokenizer.save_pretrained("./bartpho_lora_final")
print("Đã lưu adapter LoRA tại ./bartpho_lora_final")
