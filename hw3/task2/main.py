import re
from openai import OpenAI
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-b17cdda9148c47ec950a97af2d93c4f2"
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 加载gsm8k数据集的测试部分
dataset = load_dataset("gsm8k", "socratic", split="test")

original_dataset = dataset

# 只评测500条数据
dataset = dataset.select(range(500))

correct_count = 0

def extract_final_answer(answer):
    match = re.search(r'####\s*(-?\d+)', answer)
    if match:
        return match.group(1)
    return None

def zero_shot_prompt(question):
    '''
    return message for zero-shot completion
    '''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{question}\n\nPlease provide only the final numerical answer."},
    ]
    return messages

def few_shot_prompt(question):
    '''
    return message for few-shot completion
    '''
    num_examples = 4
    examples = original_dataset.select(range(len(original_dataset) - num_examples, len(original_dataset)))
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{examples[0]['question']}\n\nPlease provide only the final numerical answer."},
        {"role": "assistant", "content": f"{extract_final_answer(examples[0]['answer'])}"},
        {"role": "user", "content": f"{examples[1]['question']}\n\nPlease provide only the final numerical answer."},
        {"role": "assistant", "content": f"{extract_final_answer(examples[1]['answer'])}"},
        {"role": "user", "content": f"{examples[2]['question']}\n\nPlease provide only the final numerical answer."},
        {"role": "assistant", "content": f"{extract_final_answer(examples[2]['answer'])}"},
        {"role": "user", "content": f"{examples[3]['question']}\n\nPlease provide only the final numerical answer."},
        {"role": "assistant", "content": f"{extract_final_answer(examples[3]['answer'])}"},
        {"role": "user", "content": f"{question}\n\nPlease provide only the final numerical answer."},
    ]
    return messages

def cot_zero_shot_prompt(question):
    '''
    return message for COT zero-shot completion
    '''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{question}\n\nPlease provide a detailed step-by-step solution and conclude with the final numerical answer in the format '#### [answer]', only a single number."}
    ]
    return messages

def cot_few_shot_prompt(question):
    '''
    return message for COT few-shot completion
    '''
    num_examples = 4
    examples = original_dataset.select(range(len(original_dataset) - num_examples, len(original_dataset)))
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    
    for example in examples:
        messages.append({"role": "user", "content": f"{example["question"]}\n\nPlease provide a detailed step-by-step solution and conclude with the final numerical answer in the format '#### [answer]', only a single number."})
        messages.append({"role": "assistant", "content": f"{example["answer"]}"})
    
    messages.append({"role": "user", "content": f"{question}\n\nPlease provide a detailed step-by-step solution and conclude with the final numerical answer in the format '#### [answer]', only a single number."})
    
    return messages

def refelexion_cot_zero_shot_prompt(question):
    '''
    return message for refelexion zero-shot completion
    '''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{question}\n\nPlease provide a detailed step-by-step solution. Afterward, we suspect your answer is incorrect, identify the mistake, correct it, and provide the final answer in the format '#### [answer]', only a single number."}
    ]
    return messages
    
def evaluate_example(example, prompt_type):
    question = example["question"]
    correct_answer = extract_final_answer(example["answer"])

    if correct_answer is None:
        print(f"Failed to extract final answer from: {example['answer']}")
        return False

    if prompt_type == "zero_shot":
        messages = zero_shot_prompt(question)
    elif prompt_type == "few_shot":
        messages = few_shot_prompt(question)
    elif prompt_type == "cot_zero_shot":
        messages = cot_zero_shot_prompt(question)
    elif prompt_type == "cot_few_shot":
        messages = cot_few_shot_prompt(question)
    elif prompt_type == "refelexion_cot_zero_shot":
        messages = refelexion_cot_zero_shot_prompt(question)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
        stream=False,
    )

    generated_answer = response.choices[0].message.content.strip()
    if not generated_answer.isdigit():
        generated_answer = extract_final_answer(generated_answer)
    
    return generated_answer == correct_answer

prompt_types = ["refelexion_cot_zero_shot"]

if __name__ == "__main__":
    # for prompt_type in prompt_types:
    #     correct_count = 0
    #     with ThreadPoolExecutor(max_workers=20) as executor:
    #         futures = [executor.submit(evaluate_example, example, prompt_type) for example in dataset]
    #         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {prompt_type}"):
    #             if future.result():
    #                 correct_count += 1

    #     accuracy = correct_count / len(dataset) * 100
    #     print(f"Accuracy for {prompt_type}: {accuracy:.2f}%")
    # error_samples = []
    # cnt = 0
    # for idx, example in tqdm(enumerate(dataset)):
    #     if not evaluate_example(example, "zero_shot"):
    #         error_samples.append((idx, example["question"], example["answer"]))
    #         cnt += 1
    #     if cnt == 10:
    #         break

    # for idx, question, answer in error_samples:
    #     print(f"Error Sample {idx}:\nQuestion: {question}\nAnswer: {answer}\n")

    dataset = dataset.select([2])

    for prompt_type in prompt_types:
        print(f"Results for {prompt_type}:")
        for idx, example in enumerate(dataset):
            question = example["question"]
            # correct_answer = extract_final_answer(example["answer"])
            correct_answer = example["answer"]
            messages = None

            if prompt_type == "zero_shot":
                messages = zero_shot_prompt(question)
            elif prompt_type == "few_shot":
                messages = few_shot_prompt(question)
            elif prompt_type == "cot_zero_shot":
                messages = cot_zero_shot_prompt(question)
            elif prompt_type == "cot_few_shot":
                messages = cot_few_shot_prompt(question)
            elif prompt_type == "refelexion_cot_zero_shot":
                messages = refelexion_cot_zero_shot_prompt(question)

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0,
                stream=False,
            )

            generated_answer = response.choices[0].message.content.strip()
            # if not generated_answer.isdigit():
            #     generated_answer = extract_final_answer(generated_answer)

            print(f"Message: {messages}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Correct Answer: {correct_answer}")
            print()