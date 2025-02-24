from datasets import load_dataset

def custom_load_dataset(load_dir):
    dataset = load_dataset(load_dir)
    
    def convert_to_prompt_only(example):

        if "<|im_start|>assistant\n<think>" in example["prompt"][0]['content']:
            user_content = example["prompt"][0]['content'].split("<|im_start|>assistant\n<think>")[0]
            prompt = [{"role": "user", "content": user_content}, {"role": "assistant", "content": "<|im_start|>assistant\n<think>"}]
        else:
            prompt = [{"role": "user", "content": example["prompt"][0]['content']}]
        gt = example["reward_model"]['ground_truth']
        return {"prompt": prompt, "ground_truth": gt, "data_source": example["data_source"]}

    column_names = dataset['train'].column_names

    dataset = dataset.map(
        convert_to_prompt_only,
        remove_columns=column_names,
        num_proc=1,
    )

    return dataset
    
