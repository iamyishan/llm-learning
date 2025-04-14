import json

from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
#model_name = 'unsloth/Qwen2.5-7B'
model_name = r"E:\develop-workspace\.cache\huggingface\Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("-----tokenizer加载成功----")
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
print("-----模型加载成功----")

# 第二步制作数据集
from data_prepare import samples

with open("datasets.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line = json.dumps(s, ensure_ascii=False)
        f.write(json_line + '\n')
    else:
        print("prepare data finished")

# 第三步，准备训练集和测试集
from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": "datasets.jsonl"}, split="train")
print("数据量：", len(dataset))

train_test_split = dataset.train_test_split(test_size=0.1)  # 45:5
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print("train dataset len:", len(train_dataset))
print("test dataset len:", len(eval_dataset))

print("------完成训练数据的准备工作------------")


# 第四步：编写tokennizer处理工具

def tokenizer_function(many_samples):
    texts = [f"{prompt}\n{completion}" for prompt, completion in
             zip(many_samples["prompt"], many_samples["completion"])]
    tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_train_dataset = train_dataset.map(tokenizer_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenizer_function, batched=True)

print("---------完成tokenizing-----------")
# print(tokenized_train_dataset[0])
# 第五步：量化设置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
print("--------已经完成量化模型的加载----------")

# 第六步：lora微调设置
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("----lora微调设置完毕--------")

# 第七步：设置训练参数
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./finetuned_models",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune"
)
print("--------训练参数设置完毕---------")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

print("-----开始训练---------")
trainer.train()
print("-----训练完成√√√√---------")

if __name__ == '__main__':
    pass
