from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

max_prompt_length = 400
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
model_name = "../../Qwen2.5-0.5B-Instruct/"
assert os.path.isdir(model_name), f"Model {model_name} does not exist"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

from verl.utils.reward_score.kk import compute_score
from verl.workers.reward_manager import NaiveRewardManager

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

from verl.utils.dataset.rl_dataset import RLHFDataset
train_files = "../Logic-RL/data/kk/instruct/3ppl/train.parquet"

dataset = RLHFDataset(parquet_files=train_files,
                      tokenizer=tokenizer,
                      prompt_key="prompt",
                      max_prompt_length=max_prompt_length,
                      filter_prompts=True,
                      return_raw_chat=True,
                      truncation='error')

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        compute_score,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()