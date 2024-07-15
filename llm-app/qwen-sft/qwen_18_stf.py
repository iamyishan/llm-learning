import os
import time

command = """
python ./Qwen/finetune.py \
--model_name_or_path "E:/develop-workspace/model_repository/modelscope/qwen/Qwen-1_8B-Chat-Int4" \
--data_path "./train_data.json" \
--fp16 True \
--output_dir output_qwen \
--num_train_epochs 10 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 10 \
--learning_rate 3e-4 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 512 \
--lazy_preprocess True \
--gradient_checkpointing True \
--use_lora True
"""

if __name__ == '__main__':
    start_time = time.time()
    os.system(command)
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
