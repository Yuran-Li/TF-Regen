import json
import math
import matplotlib.pyplot as plt
from .check import check_answer_math, check_answer_gsm8k, check_answer_gsm_hard, check_answer_strategyqa, check_answer_coin, check_answer_legal, check_answer_letter, extract_answer, check_answer_bamboogle, check_answer_Headline

task_config = {
        "gsm8k": ("answer", check_answer_gsm8k),
        "gsm_hard": ("target", check_answer_gsm_hard),
        "strategyQA": ("answer", check_answer_strategyqa),
        "bamboogle": ("Answer", check_answer_bamboogle),
        "math": ("answer", check_answer_math),
        "coin": ("targets", check_answer_coin),
        "legal": ("answer", check_answer_legal),
        "letter": ("answer", check_answer_letter),
        "Headline": ("Price Sentiment", check_answer_Headline)
    }

def shannon_entropy(frequencies):
    total_responses = sum(frequencies)
    if total_responses == 0:
        return 0  
    
    entropy = 0
    for freq in frequencies:
        if freq > 0:
            p_i = freq / total_responses  
            entropy -= p_i * math.log(p_i, 2)  
    return entropy

def evaluate_task(data, task):
    """
    according to the task, evaluate the accuracy of CoT answer and label answer
    """
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    label_field, check_func = task_config[task]
    generations = [item['rectify_CoT'] if 'rectify_CoT' in item else (item['CoT'][0] if isinstance(item['CoT'], list) and len(item['CoT']) > 0 else item['CoT']) for item in data]
    labels = [item[label_field] for item in data]
    
    total_questions = len(generations)
    correct_answers = 0
    
    print(f"total questions: {total_questions}")
    print("-" * 50)
    
    for generation, label in zip(generations, labels):
        is_correct = check_func(generation, label)
        if is_correct:
            correct_answers += 1
    
    accuracy = correct_answers / total_questions * 100
    
    return accuracy, correct_answers, total_questions

def evaluate(task: str, json_file: str):
    """
    main evaluation function, load data and call the unified evaluation function
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return evaluate_task(data, task)

def evaluate_uncertainty(task: str, json_file: str, args):
    """
    evaluate the accuracy of multiple runs
    """
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    label_field, check_func = task_config[task]
    load_data = json.load(open(json_file, 'r', encoding='utf-8'))
    count_correct = 0
    for item in load_data:
        cot_list = item["CoT"]  # save the original CoT list
        majority_vote = 0

        # calculate the majority vote
        for run_idx in range(len(cot_list)):
            # print("run_idx", run_idx)
            current_cot = cot_list[run_idx]
            is_correct = check_func(current_cot, item[label_field])
            if is_correct:
                majority_vote += 1
                
        if majority_vote > len(cot_list) / 2:
            item["majority_vote"] = True
            count_correct += 1
        else:
            item["majority_vote"] = False

        ## calculate the shannon entropy
        # count the frequency of each answer
        answer_counts = {}
        for run_idx in range(len(cot_list)):
            answer = extract_answer(cot_list[run_idx])
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # convert to frequency list
        frequencies = list(answer_counts.values())
        entropy = shannon_entropy(frequencies)
        item["entropy"] = entropy
    accuracy = count_correct / len(load_data) * 100

    with open(f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation({args.method})_uncertainty_analysis.json", "w", encoding="utf-8") as f:
        json.dump(load_data, f, ensure_ascii=False, indent=2)

    ##analyze the relation between entropy and accuracy
    from collections import defaultdict
    import numpy as np
    
    # collect entropy values and corresponding majority vote results
    entropy_correct = defaultdict(int)
    entropy_incorrect = defaultdict(int)
    
    for item in load_data:
        entropy = round(item["entropy"], 2)  # round to 2 decimal places for grouping
        if item["majority_vote"]:
            entropy_correct[entropy] += 1
        else:
            entropy_incorrect[entropy] += 1
    
    # get all unique entropy values and sort them
    all_entropies = sorted(set(list(entropy_correct.keys()) + list(entropy_incorrect.keys())))
    
    # prepare data for plotting
    correct_counts = [entropy_correct[e] for e in all_entropies]
    incorrect_counts = [entropy_incorrect[e] for e in all_entropies]
    
    # create bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(all_entropies))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, correct_counts, width, label='correct', color='#87CEEB', alpha=0.8)
    bars2 = plt.bar(x + width/2, incorrect_counts, width, label='incorrect', color='#F08080', alpha=0.8)
    
    # calculate total samples for percentage calculation
    total_samples = len(load_data)
    
    # add percentage labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if correct_counts[i] > 0:
            percentage1 = (correct_counts[i] / total_samples) * 100
            plt.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                    f'{percentage1:.1f}%', ha='center', va='bottom', fontsize=8)
        
        if incorrect_counts[i] > 0:
            percentage2 = (incorrect_counts[i] / total_samples) * 100
            plt.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1,
                    f'{percentage2:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel("Entropy")
    plt.ylabel("number of samples")
    plt.title(f"Entropy corresponding to correctness({len(load_data)} samples)")
    plt.xticks(x, [f"{e:.2f}" for e in all_entropies], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"output/{args.model_name}_{args.model_type}_{args.dataset_name}_{args.split}_CoT_generation({args.method})_uncertainty_analysis.pdf")
    plt.close()

    return accuracy, count_correct, len(load_data)

def extract_final_answer(text):
    """
    extract the content after 'Final Answer:'
    """
    if "Final Answer:" in text:
        # use split to get the part after Final Answer:
        answer = text.split("Final Answer:")[1].strip()
        return answer
    else:
        return None

def evaluate_AoT(task: str, json_file: str):
    """
    evaluate the accuracy of AoT
    """
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    label_field, check_func = task_config[task]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    correct_answers = 0
    
    print(f"total questions: {total_questions}")
    print("-" * 50)
    
    for item in data:
        prediction = extract_final_answer(item["AoT"])
        if prediction is not None:
            is_correct = check_func("####" + prediction, item[label_field])
            if is_correct:
                correct_answers += 1
        
    accuracy = correct_answers / total_questions * 100
    return accuracy, correct_answers, total_questions

def evaluate_reflexion(task: str, json_file: str):
    """
    evaluate the accuracy of Reflexion
    """
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    label_field, check_func = task_config[task]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    correct_answers = 0
    
    print(f"total questions: {total_questions}")
    print("-" * 50)
    
    for item in data:
        # Recompute correctness against GT to avoid contamination by judge noise
        generation = item["reflexion"]["answer"]
        label = item[label_field]
        is_correct = check_func(generation, label)
        if is_correct:
            correct_answers += 1
        
    accuracy = correct_answers / total_questions * 100
    return accuracy, correct_answers, total_questions

def evaluate_self_refine(task: str, json_file: str):
    """
    evaluate the accuracy of Self-Refine
    """
    if task not in task_config:
        raise ValueError(f"Unsupported task: {task}")
    
    label_field, check_func = task_config[task]
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    correct_answers = 0
    
    print(f"total questions: {total_questions}")
    print("-" * 50)
    
    for item in data:
        correct = item["self_refine"]["correct"]
        if correct:
            correct_answers += 1
        
    accuracy = correct_answers / total_questions * 100
    return accuracy, correct_answers, total_questions

if __name__ == "__main__":
    # evaluate the accuracy
    file_path = "./output/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo_api_coin_test_CoT_generation(zero_shot).json"
    accuracy, correct, total = evaluate(task="coin", json_file=file_path)
    print(f"accuracy: {accuracy:.2f}%")
    print(f"correct: {correct}")
    print(f"total: {total}")