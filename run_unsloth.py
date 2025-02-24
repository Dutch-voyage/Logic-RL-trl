from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

max_prompt_length = 400
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
model_name = "/data1/yyx/models/Qwen2.5-0.5B/"
assert os.path.isdir("/data1")
assert os.path.isdir(model_name), f"Model {model_name} does not exist"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 1,
)


from trl.trainer import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = False, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 10,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

from utils.dataset_utils import custom_load_dataset
from utils.kk import compute_score


def _select_rm_score_fn(data_source):
    return compute_score

class RewardManager():
    """The reward manager.
    """

    def __init__(self, num_examine) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, prompts, completions, **kwargs):
        """We will expand this function gradually based on the available datasets"""

        rewards = []
        already_print_data_sources = {}

        for i in range(len(completions)):
            sequences_str = completions[i][0]['content']

            ground_truth = kwargs['ground_truth'][i]

            # select rm_score
            data_source = kwargs['data_source'][0]
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            rewards.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return rewards
    
    def logging(self, reward_tensor):
        reward_metrics = {}
        reward_metrics["reward/mean"] = torch.mean(reward_tensor).detach().item()
        # Calculate all_correct ratio (value == 3)
        all_correct = torch.sum(reward_tensor == 3).float() / reward_tensor.numel()
        reward_metrics["reward/all_correct_ratio"] = all_correct.detach().item()
        # Calculate format_error ratio (value == -1)
        format_error = torch.sum(reward_tensor == -1).float() / reward_tensor.numel()
        reward_metrics["reward/format_error_ratio"] = format_error.detach().item()
        # Calculate wrong answer ratio (value == -1)
        format_error = torch.sum(reward_tensor == -0.5).float() / reward_tensor.numel()
        reward_metrics["reward/wrong_answer_ratio"] = format_error.detach().item()
        
        return reward_metrics

data_dir = "../Logic-RL/data/kk/instruct/3ppl/"

dataset = custom_load_dataset(data_dir)

reward_manager = RewardManager(num_examine=1)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [    
        reward_manager,
    ],
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
)
trainer.train()