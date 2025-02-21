from datasets import load_dataset

def load_dataset():
    dataset = load_dataset("../Logic-RL/data/kk/instruct/3ppl/")
    
    def convert_to_prompt_only(example):
        prompt = [{"role": "user", "content": example["prompt"][0]['content']}]
        label = example["reward_model"]['ground_truth']
        return {"prompt": prompt, "label": label}

    column_names = dataset['train'].column_names

    dataset = dataset.map(
        convert_to_prompt_only,
        remove_columns=column_names,
        num_proc=1,
    )

    return dataset
    
