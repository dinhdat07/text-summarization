# %% [markdown]
# # Notebook 6: LoRA Fine-Tuning Qwen2.5-0.5B (Long-Context)
#
# Fine-tune Qwen2.5-0.5B cho bài toán Text Summarization.
# ĐIỂM NHẤN: Qwen hỗ trợ 32k context length. Ta sẽ set max_length = 2048 để xử lý nguyên khối các bài báo dài,
# và dùng Gradient Checkpointing để không bị Out-Of-Memory trên GPU T4.

# %%
# !pip install -q transformers datasets peft bitsandbytes accelerate "torchao>=0.16.0"

# %%
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd

# %% [markdown]
# ## 1. Load Data
# Chỉ load tập train và validation từ file CSV đã chuẩn bị ở Notebook 1.

# %%
train_df = pd.read_csv("train_10k.csv")
val_df = pd.read_csv("val_1k.csv")

from datasets import Dataset, DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df)
})

# %% [markdown]
# ## 2. Tokenizer & Model

# %%
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tải model với FP16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,

    trust_remote_code=True
)

# Bật Gradient Checkpointing để tiết kiệm VRAM cho long-context
model.gradient_checkpointing_enable()

# %% [markdown]
# ## 3. Prepare Dataset cho ngữ cảnh dài (2048 tokens)

# %%
def format_prompt(example):
    text = f"Tóm tắt bài báo sau:\n{example['article']}\n\nTóm tắt:\n{example['abstract']}{tokenizer.eos_token}"
    return {"text": text}

formatted_dataset = dataset.map(format_prompt)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048, # Tăng max_length lên 2048 (gấp đôi BARTpho)
        padding="max_length"
    )

tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names + ["text"])
data_collator = DataCollatorForLanguageModeling(processing_class=tokenizer, mlm=False)

# %% [markdown]
# ## 4. Cấu hình LoRA

# %%
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 5. Training

# %%
training_args = TrainingArguments(
    output_dir="./qwen_lora_results",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=1, # Phải giảm batch_size xuống 1 vì max_length=2048 rất tốn RAM
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True, 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

trainer.train()

# %% [markdown]
# ## 6. Lưu mô hình

# %%
trainer.save_model("./qwen-lora-vietnews")
tokenizer.save_pretrained("./qwen-lora-vietnews")
print("Đã lưu LoRA adapter!")
