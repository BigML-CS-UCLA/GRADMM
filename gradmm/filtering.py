import os
import re
import glob
import json
import copy
import argparse
from collections import defaultdict

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from datasets import load_dataset
from tqdm import tqdm

from data_utils import TextDataset
from utilities import cos_sim, compute_grads_lm


PROMPT_TEMPLATE = {
    "sst2": "Given the following movie review. Only output {pos_label} or {neg_label}.",
    "rotten_tomatoes": "Given the following movie review. Only output {pos_label} or {neg_label}.",
    "TwitterEmotion": "Given the following tweet. Only output {pos_label} or {neg_label}."
}


FEWSHOT_DICT = {
    "sst2": [
        ("The movie was fantastic and thrilling!", "{pos_label}"),
        ("I hated the film; it was boring and slow.", "{neg_label}"),
        ("What a masterpiece, truly inspiring!", "{pos_label}"),
        ("The plot was dull and characters uninspiring.", "{neg_label}"),
    ],
    "rotten_tomatoes": [
        ("The movie was fantastic and thrilling!", "{pos_label}"),
        ("I hated the film; it was boring and slow.", "{neg_label}"),
        ("What a masterpiece, truly inspiring!", "{pos_label}"),
        ("The plot was dull and characters uninspiring.", "{neg_label}"),
    ],
    "TwitterEmotion": [
        ("i feel lucky to know what its like to revel in the freedom and wide open spaces that being by the sea affords", "{pos_label}"),
        ("i feel depressed or even short tempered some days", "{neg_label}"),
        ("i said i feel incredibly thankful on the whole", "{pos_label}"),
        ("i left feeling defeated like nothing had been accomplished the day a complete waste of time amp energy", "{neg_label}"),
    ]
}


MODEL_MAP = {
    "phi": "microsoft/phi-1_5",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def get_args(argv=None):
    """Parse arguments.

    Returns:
    """
    parser = argparse.ArgumentParser(description='Filtering')
    parser.add_argument('--model_name', type=str, default='phi')
    parser.add_argument(
        '--dataset',
        choices=['sst2', 'rotten_tomatoes', 'TwitterEmotion'],
        required=True,
    )
    parser.add_argument(
        '--split', choices=['train', 'validation'], default='train'
    )
    parser.add_argument(
        '--gen_bs', type=int, default=1
    )   # number of embeddings to generate per time
    parser.add_argument('--n_gen_samples', type=int, default=100)
    parser.add_argument('--n_fewshot', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--pos_label', type=str, default='positive')
    parser.add_argument('--neg_label', type=str, default='negative')
    parser.add_argument(
        '--use_instruction',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--use_fewshot',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--filter_score', choices=['cls'], default='cls'
    )
    parser.add_argument(
        '--filter_method', choices=['remove', 'relabel', 'top_score', 'bottom_score', 'first', 'greedy_selection'], default='remove'
    )
    parser.add_argument(
        '--file_dir',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
    )
    parser.add_argument('--coeff_perplexity', type=float, default=1)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument(
        '--clean',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--balance_score',
        type=str2bool,
        nargs='?',
        default=False,
    )
    parser.add_argument(
        '--per_label',
        type=str2bool,
        nargs='?',
        default=True,
    )
    parser.add_argument(
        '--interleave_label',
        type=str2bool,
        nargs='?',
        default=False,
    )
    
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv[1:])
    return args


def construct_prompt(dataset, pos_label, neg_label):
    return PROMPT_TEMPLATE[dataset].format_map({
        "pos_label": pos_label, "neg_label": neg_label
    })


# Few-shot prompting function
def few_shot_classification(input_text, labels, model, tokenizer, instruction_prompt=None, examples=None):
    """
    Perform few-shot classification using a decoder-only language model.

    Args:
        input_text (str): Input text for classification.
        labels (list): List of possible labels (e.g., ["positive", "negative"]).
        examples (list): List of tuples containing few-shot examples as (text, label).

    Returns:
        dict: Dictionary with labels and their normalized scores.
    """
    # Build the few-shot prompt
    if instruction_prompt is None:
        prompt = ""
    else:
        prompt = instruction_prompt
    if examples is not None:
        for example_text, example_label in examples:
            example_label = example_label.format_map({
                "pos_label": labels[0], 
                "neg_label": labels[1]
            })
            prompt += f"Input: {example_text}\nLabel: {example_label}\n\n"
    prompt += f"Input: {input_text}\nLabel:"
    # print(prompt)

    # Calculate label probabilities
    label_scores = {}
    
    for label in labels:
        # Append label to prompt

        label_prompt = f"{prompt} {label}"
        inputs = tokenizer(label_prompt, return_tensors="pt").to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        # Get token probabilities for the label
        label_tokens = tokenizer(label, return_tensors="pt")["input_ids"].to(model.device)
        label_logits = shifted_logits[0, -len(label_tokens[0]) :, :]
        probabilities = torch.nn.functional.softmax(label_logits, dim=-1)

        # Compute the probability of the label
        score = torch.prod(
            probabilities[torch.arange(len(label_tokens[0])), label_tokens[0]]
        ).item()
        label_scores[label] = score
    # Normalize scores to sum to 1
    total_score = sum(label_scores.values())
    label_scores = {k: v / total_score for k, v in label_scores.items()}

    return label_scores


def get_first_element(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def filter_synthetic_data(args, file_path, model, tokenizer):
    # Read the first JSONL file
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = json.loads(line)
            samples.append(sample)
    
    # Filter out wrong label
    filtered_samples = []
    labels = [args.pos_label, args.neg_label]
    if args.use_instruction:
        instruction_prompt = construct_prompt(
            args.dataset,
            args.pos_label,
            args.neg_label
        )
    else:
        instruction_prompt = None
    if args.use_fewshot:
        few_shot_examples = FEWSHOT_DICT[args.dataset]
    else:
        few_shot_examples = []
    
    mis_match_count = 0
    for sample in tqdm(samples):
        seq = get_first_element(sample["inputs"])
        if args.dataset == "sst2":
            seq = seq.split("It was")[0].strip()
        if args.clean:
            seq = clean_text(seq)
        syn_label = sample["label"]
        if args.filter_score == "cls":
            score_dict = few_shot_classification(
                seq,
                labels,
                model,
                tokenizer,
                instruction_prompt=instruction_prompt,
                examples=few_shot_examples
            )
        else:
            raise ValueError(f"{args.filter_score} is not supported.")
        pos_score = score_dict[args.pos_label]
        neg_score = score_dict[args.neg_label]
        if pos_score > neg_score:
            model_label = 1
        elif pos_score < neg_score:
            model_label = 0
        else:
            model_label = "undefined"
        
        if syn_label == model_label:
            filtered_samples.append(sample)
        else:
            # print(seq)
            # print(f"Mismatch labels: model label = {model_label} and synthetic label = {syn_label}")
            mis_match_count += 1
            if args.filter_method == "relabel" and model_label != "undefined":
                sample["label"] = model_label
                filtered_samples.append(sample)
    
    print(f"Mismatch count = {mis_match_count}")        
    
    return filtered_samples


def interleave_label(list_sample):
    """Assume only 2 different labels."""
    grouped_samples = defaultdict(list)
    for sample in list_sample:
        grouped_samples[sample['label']].append(sample)
    
    # Count
    labels = list(grouped_samples.keys())
    count_label0 = len(grouped_samples[labels[0]])
    count_label1 = len(grouped_samples[labels[1]])
    total_count = count_label0 + count_label1
    min_count = min(count_label0, count_label1)
    pivot_idx = 2 * min_count
    mixed_samples = [None for _ in range(total_count)]
    
    # Mix samples
    mixed_samples[:pivot_idx:2] = grouped_samples[labels[0]][:min_count]
    mixed_samples[1:pivot_idx:2] = grouped_samples[labels[1]][:min_count]
    # Handle imbalance case
    remaining = grouped_samples[labels[0]][min_count:] + grouped_samples[labels[1]][min_count:]
    mixed_samples[pivot_idx:] = remaining
    
    return mixed_samples


def extract_top_samples_per_label(file_path, alpha, top_n=100, per_label=True, interleave=False, reverse=False, balance_score=False):
    """
    Extracts the top N samples for each label from a JSONL file based on the metric:
    rec_loss_ids + alpha * perplexity.

    Parameters:
        file_path (str): Path to the JSONL file.
        alpha (float): Multiplier for the perplexity term in the metric.
        top_n (int): Number of top samples to extract per label.

    Returns:
        dict: A dictionary where keys are labels, and values are lists of top N samples.
    """
    # Read the JSONL file
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = json.loads(line)
            samples.append(sample)
    
    # Group samples by label
    grouped_samples = defaultdict(list)
    for sample in samples:
        if per_label:
            grouped_samples[sample['label']].append(sample)
        else:
            grouped_samples[0].append(sample)
    
    # Compute the metric and select top samples per label
    # Compute the metric, remove duplicates, and select top samples per label
    top_samples_per_label = []
    
    for label, label_samples in grouped_samples.items():
        # Remove duplicate sentences
        unique_samples = {get_first_element(sample["inputs"]): sample for sample in label_samples}.values()
        
        print(len(unique_samples))

        # Compute the metric
        for sample in unique_samples:
            sample['metric'] = sample['rec_loss_ids'] + alpha * sample['perplexity']

        # Sort samples by the metric
        sorted_samples = sorted(unique_samples, key=lambda x: x['metric'], reverse=reverse)

        # Select the top N samples
        top_samples_per_label.extend(sorted_samples[:top_n])
        
    if interleave:
        top_samples_per_label = interleave_label(top_samples_per_label)
        
    if balance_score:
        top_samples_per_label = balance_classes_by_average_score(top_samples_per_label, reverse=reverse)

    return top_samples_per_label


def balance_classes_by_total_score(samples):
    """
    Removes samples from one class to make the total scores of the two classes equal.

    Parameters:
        samples (list): A list of dictionaries where each dictionary has 'metric' (score) and 'label'.

    Returns:
        list: A balanced list of samples with equal total scores for the two labels.
    """
    # Separate samples by label
    label_0_samples = [s for s in samples if s['label'] == 0]
    label_1_samples = [s for s in samples if s['label'] == 1]
    
    # Compute total scores for each label
    total_score_0 = sum(s['metric'] for s in label_0_samples)
    total_score_1 = sum(s['metric'] for s in label_1_samples)
    
    # Determine which label has the excess score
    if total_score_0 > total_score_1:
        excess_samples = label_0_samples
        target_samples = label_1_samples
        excess_score = total_score_0 - total_score_1
    else:
        excess_samples = label_1_samples
        target_samples = label_0_samples
        excess_score = total_score_1 - total_score_0

    # Sort excess samples by score (ascending order)
    excess_samples.sort(key=lambda x: x['metric'])
    
    # Remove samples to balance the scores
    score_removed = 0
    balanced_excess_samples = []
    
    # Sample are sorted by score in ascending order
    for sample in excess_samples[::-1]:
        if score_removed > excess_score:
            balanced_excess_samples.append(sample)
        score_removed += sample['metric']
    
    # Combine the balanced samples
    balanced_samples = target_samples + balanced_excess_samples

    return balanced_samples


def balance_classes_by_average_score(samples, reverse=False):
    """
    Removes samples from one class to make the average scores of the two classes equal.

    Parameters:
        samples (list): A list of dictionaries where each dictionary has 'metric' (score) and 'label'.

    Returns:
        list: A balanced list of samples with equal average scores for the two labels.
    """
    print("Start balancing")
    # Separate samples by label
    label_0_samples = [s for s in samples if s['label'] == 0]
    label_1_samples = [s for s in samples if s['label'] == 1]

    # Compute total scores and counts for each label
    total_score_0 = sum(s['metric'] for s in label_0_samples)
    total_score_1 = sum(s['metric'] for s in label_1_samples)
    count_0 = len(label_0_samples)
    count_1 = len(label_1_samples)

    # Calculate the current averages
    avg_score_0 = total_score_0 / count_0 if count_0 > 0 else 0
    avg_score_1 = total_score_1 / count_1 if count_1 > 0 else 0

    # Determine which label has the higher average score
    if (avg_score_0 > avg_score_1 and not reverse) or (avg_score_0 < avg_score_1 and reverse):
        print(f"Label 0 has a higher average score of {avg_score_0} compared to {avg_score_1}")
        excess_samples = label_0_samples
        target_samples = label_1_samples
        target_avg = avg_score_1
        total_excess = total_score_0
        count_excess = count_0
    else:
        print(f"Label 1 has a higher average score of {avg_score_1} compared to {avg_score_0}")
        excess_samples = label_1_samples
        target_samples = label_0_samples
        target_avg = avg_score_0
        total_excess = total_score_1
        count_excess = count_1

    # Sort excess samples by score (descending order for impactful removal)
    excess_samples.sort(key=lambda x: x['metric'], reverse=not reverse)

    # Remove samples to balance the average scores
    while len(excess_samples) > 0:
        # Remove the highest-score sample from the excess samples
        sample = excess_samples.pop(0)
        
        total_excess -= sample['metric']
        count_excess -= 1
        current_avg_excess = total_excess / count_excess if count_excess > 0 else 0

        # Stop if the averages are balanced
        if (current_avg_excess <= target_avg and not reverse) or (current_avg_excess >= target_avg and reverse):
            break

    # Combine the balanced samples
    balanced_samples = target_samples + excess_samples
    print("Finish balancing")

    return balanced_samples


def extract_first_samples_per_label(file_path, top_n=100, start_idx=0, per_label=True, interleave=False):
    """
    Extracts the first N samples for each label from a JSONL file.

    Parameters:
        file_path (str): Path to the JSONL file.
        top_n (int): Number of top samples to extract per label.

    Returns:
        dict: A dictionary where keys are labels, and values are lists of top N samples.
    """
    # Read the JSONL file
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = json.loads(line)
            samples.append(sample)
    
    # Group samples by label
    grouped_samples = defaultdict(list)
    for sample in samples:
        if per_label:
            grouped_samples[sample['label']].append(sample)
        else:
            grouped_samples[0].append(sample)
    
    # Compute the metric and select top samples per label
    # Compute the metric, remove duplicates, and select top samples per label
    top_samples_per_label = []
    
    for label, label_samples in grouped_samples.items():
        # Remove duplicate sentences
        unique_samples = {get_first_element(sample["inputs"]): sample for sample in label_samples}.values()
        
        print(len(unique_samples))

        # Select the top N samples
        top_samples_per_label.extend(list(unique_samples)[start_idx:start_idx+top_n])
        
    if interleave:
        top_samples_per_label = interleave_label(top_samples_per_label)

    return top_samples_per_label


def greedy_selection(args, file_path, model, tokenizer, real_pos_grads, real_neg_grads, previous_pos_grads=None, previous_neg_grads=None, deter_usm=True):
    if previous_pos_grads is not None:
        new_json_file = args.json_file + f'_cond'
    else:
        new_json_file = args.json_file
    # Load synthetic data
    syn_pos_sequences, syn_neg_sequences = load_syn_data(file_path, args.dataset)
    syn_pos_labels = [torch.tensor([1]) for _ in range(len(syn_pos_sequences))]
    syn_neg_labels = [torch.tensor([0]) for _ in range(len(syn_neg_sequences))]
    
    # Greedy selection
    list_greedy_pos_idx = greedy_grad_selection(
        args,
        model, 
        tokenizer, 
        syn_pos_sequences, 
        syn_pos_labels, 
        real_pos_grads, 
        top_n=args.top_n,
        best_previous_grads=previous_pos_grads
    )
    
    list_greedy_neg_idx = greedy_grad_selection(
        args,
        model, 
        tokenizer, 
        syn_neg_sequences, 
        syn_neg_labels, 
        real_neg_grads, 
        top_n=args.top_n,
        best_previous_grads=previous_neg_grads
    )
    
    grouped_idx = {
        0: list_greedy_neg_idx,
        1: list_greedy_pos_idx
    }
    
    # Read the JSONL file
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = json.loads(line)
            samples.append(sample)
    
    # Group samples by label
    grouped_samples = defaultdict(list)
    for sample in samples:
        grouped_samples[sample['label']].append(sample)
    
    top_samples_per_label = []
    
    for label, label_samples in grouped_samples.items():
        # Select the top N samples
        for idx in grouped_idx[label]:
            top_samples_per_label.append(label_samples[idx])
        
    if args.interleave_label:
        top_samples_per_label = interleave_label(top_samples_per_label)
    
    if deter_usm:
        new_json_file += f'_deter_usm_top{args.top_n}'
    else:
        new_json_file += f'_greedy_top{args.top_n}'
    if args.interleave_label:
        new_json_file += '_interleave'

    return top_samples_per_label, new_json_file


def grad_dist(target_grads, curr_grads, tag_factor=1, metric="cos"):
    ret = 0.0
    n_g = 0
    
    for g1, g2 in zip(target_grads, curr_grads):
        if (g1 is not None) and (g2 is not None):
            if metric == 'cos':
                ret += 1 - cos_sim(g1, g2)
            elif metric == 'dlg':
                ret += (g1 - g2).square().sum()
            elif metric == 'tag':
                ret += (g1 - g2).square().sum() + tag_factor * torch.abs(
                    g1 - g2
                ).sum()
            n_g += 1
    
    if metric == 'cos':
        ret /= n_g
    
    return ret


def compute_grads(args, model, tokenizer, sequences, labels, aggregate=True, agg_dim=0):
    """Compute average gradients.

    Args:
        args: Arguments
        model: Model
        tokenizer: Tokenizer
        sequences: List of samples
        labels: Labels

    Returns:
        average_grads: Average gradients
        list_true_embeds: List of true embeddings
        avg_true_embeds: Average true embeddings
        closest_index: Index of the gradient closest to the average gradient
        prompt_lengths: List of prompt lengths
    """
    lm_embeddings = model.get_input_embeddings()
    num_samples = len(sequences)
    text_labels = []
    prompt_lengths = []  # Collect prompt lengths for dataset with two prompts
    if args.dataset in ["sst2", "rotten_tomatoes"]:
        sequences = [seq + " It was " for seq in sequences]
        for seq in sequences:
            prompt_len = len(
                tokenizer(seq)["input_ids"]
            )  # Total token count for sst2 prompt + sequence
            prompt_lengths.append(prompt_len)
        for label in labels:
            text_labels.append("bad" if label.flatten() == 0 else "great")
    elif args.dataset == "TwitterEmotion":
        sequences = [
            seq + " Does the tweet express joy or sadness?\n" for seq in sequences
        ]
        for seq in sequences:
            prompt_len = len(
                tokenizer(seq)["input_ids"]
            )  # Total token count for TwitterEmotion prompt + sequence
            prompt_lengths.append(prompt_len)
        for label in labels:
            text_labels.append("sadness" if label.flatten() == 0 else "joy")
    
    list_grads = []

    for i in tqdm(range(num_samples)):
        seq = sequences[i]
        text_label = text_labels[i]
        orig_batch = tokenizer(
            seq, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        label = tokenizer(
            text_label, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        label = label["input_ids"].view(-1)
        true_embeds = lm_embeddings(orig_batch["input_ids"])
        curr_grads = compute_grads_lm(
            model, true_embeds, orig_batch["attention_mask"], label
        )

        sample_grad = []
        for grad in curr_grads:
            if aggregate and len(grad.shape) > 1:
                sample_grad.append(grad.mean(dim=agg_dim).detach().clone().cpu())
            else:
                sample_grad.append(grad.detach().clone().cpu())
        del curr_grads
        list_grads.append(sample_grad)
        torch.cuda.empty_cache()

    return list_grads


def greedy_grad_selection(args, model, tokenizer, list_seq, list_label, avg_grad, best_previous_grads=None, top_n=50, deter_usm=True):
    list_syn_grads = compute_grads(
        args, 
        model, 
        tokenizer, 
        list_seq, 
        list_label, 
        aggregate=False
    )
    
    count = 0
    list_idx = list(range(len(list_syn_grads)))
    list_best_idx = []
    list_best_dist = []
    previous_grads = copy.deepcopy(best_previous_grads)
    # best_previous_grads = None

    for count in tqdm(range(top_n)):
        best_dist = 2
        best_idx = None
        # Greedy selection
        for idx in list_idx:
            syn_grads = list_syn_grads[idx]
            syn_grads = combine_grad(best_previous_grads, syn_grads, count)
            curr_dist = grad_dist(avg_grad, syn_grads, metric='cos').item()
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_idx = idx
        # print(best_idx, best_dist)
        list_best_idx.append(best_idx)
        list_best_dist.append(best_dist)
        best_previous_grads = combine_grad(best_previous_grads, list_syn_grads[best_idx], count)
        # Remove the best index
        list_idx.remove(best_idx)
        
    print("Final best distance: ", best_dist)
    
    del best_previous_grads, syn_grads
    torch.cuda.empty_cache()

    if deter_usm:
        list_best_idx = deterministic_usm(list_syn_grads, list_best_idx, avg_grad, previous_grads=previous_grads)

    del list_syn_grads
    torch.cuda.empty_cache()
        
    return list_best_idx


def grad_similarity_selection(args, file_path, model, tokenizer, real_pos_grads, real_neg_grads, previous_pos_grads, previous_neg_grads):
    assert previous_neg_grads is not None, "Please provide previous neg grad"
    assert previous_pos_grads is not None, "Please provide previous pos grad"
    
    # Load synthetic data
    syn_pos_sequences, syn_neg_sequences = load_syn_data(file_path, args.dataset)
    syn_pos_labels = [torch.tensor([1]) for _ in range(len(syn_pos_sequences))]
    syn_neg_labels = [torch.tensor([0]) for _ in range(len(syn_neg_sequences))]
    
    # Greedy selection
    list_greedy_pos_idx = grad_similarity_filtering(
        args,
        model, 
        tokenizer, 
        syn_pos_sequences, 
        syn_pos_labels, 
        real_pos_grads, 
        previous_grads=previous_pos_grads
    )
    
    list_greedy_neg_idx = grad_similarity_filtering(
        args,
        model, 
        tokenizer, 
        syn_neg_sequences, 
        syn_neg_labels, 
        real_neg_grads, 
        previous_grads=previous_neg_grads
    )
    
    grouped_idx = {
        0: list_greedy_neg_idx,
        1: list_greedy_pos_idx
    }
    
    # Read the JSONL file
    samples = []
    with open(file_path, 'r') as file:
        for line in file:
            sample = json.loads(line)
            samples.append(sample)
    
    # Group samples by label
    grouped_samples = defaultdict(list)
    for sample in samples:
        grouped_samples[sample['label']].append(sample)
    
    top_samples_per_label = []
    
    for label, label_samples in grouped_samples.items():
        # Select the top N samples
        for idx in grouped_idx[label]:
            top_samples_per_label.append(label_samples[idx])
        
    if args.interleave_label:
        top_samples_per_label = interleave_label(top_samples_per_label)
    
    new_json_file = args.json_file + f'_cond_grad_similarity'
    if args.interleave_label:
        new_json_file += '_interleave'

    return top_samples_per_label, new_json_file


def grad_similarity_filtering(args, model, tokenizer, list_seq, list_label, avg_grad, previous_grads):
    list_syn_grads = compute_grads(
        args, 
        model, 
        tokenizer, 
        list_seq, 
        list_label, 
        aggregate=False
    )
    
    count = 0
    list_idx = list(range(len(list_syn_grads)))
    list_selected_idx = []
    prev_dist = grad_dist(avg_grad, previous_grads, metric='cos').item()
    print(f"Prev dist: {prev_dist}")

    for idx in tqdm(list_idx):
        syn_grads = list_syn_grads[idx]
        syn_grads = combine_grad(previous_grads, syn_grads, count)
        curr_dist = grad_dist(avg_grad, syn_grads, metric='cos').item()
        if curr_dist <= prev_dist:
            list_selected_idx.append(idx)
    
    del syn_grads, list_syn_grads
    torch.cuda.empty_cache()
    print(f"Select new {len(list_selected_idx)} samples")
        
    return list_selected_idx


def get_output_file_name(args):
    if args.clean:
        tag = "_clean"
    else:
        tag = ""
    tag += f"_{args.filter_method}"
    tag += f"_{args.filter_score}"
    tag += f"_{args.model_name}"
    tag += f"_{args.dataset}"
    tag += f"_{args.pos_label}"
    tag += f"_{args.neg_label}"
    tag += f"_instr{args.use_instruction}"
    tag += f"_fs{args.use_fewshot}"
    
    return args.json_file + tag


def clean_text(text):
    """
    Remove special characters (\n, <|endoftext|>) from the input text.
    
    Args:
        text (str): The input string to clean.
    
    Returns:
        str: The cleaned string.
    """
    # Define a list of patterns to remove
    patterns = [r'\n', r'<\|endoftext\|>', r'\"', r'###', r'3.', r'2.']
    
    # Replace each pattern with an empty string
    text = text.replace('...', '')
    for pattern in patterns:
        text = re.sub(pattern, '', text)
        
    text = text.split("You")[0]
    text = text.split("Exercise")[0]
    
    # Optionally, strip leading and trailing spaces
    return text.strip()


def output_to_jsonl(args, list_samples, output_file, post_processing=False, num_out=None):
    mean_perplexity = 0
    mean_rec_loss = 0
    mean_rec_loss_ids = 0

    with open(output_file, 'w') as file:
        if num_out is None:
            num_out = len(list_samples)
        for sample in list_samples[:num_out]:
            if post_processing:
                input_text = get_first_element(sample["inputs"])
                if args.dataset in ["sst2", "rotten_tomatoes"]:
                    input_text = input_text.split("It was")[0].strip()
                elif args.dataset == "TwitterEmotion":
                    input_text = input_text.split("Does the tweet express joy or sadness")[0].strip()
                # print(input_text)
                sample["inputs"] = clean_text(input_text)
            file.write(json.dumps(sample) + '\n')
            mean_perplexity += sample['perplexity']
            mean_rec_loss += sample['rec_loss_embeds']
            mean_rec_loss_ids += sample['rec_loss_ids']
            
    # print(f"#samples: {len(list_samples)}")
    # print(f"mean_perplexity: {mean_perplexity / len(list_samples)}")
    # print(f"mean_rec_loss: {mean_rec_loss / len(list_samples)}")
    # print(f"mean_rec_loss_ids: {mean_rec_loss_ids / len(list_samples)}")
    
    
def load_real_data(dataset_name, split, device, n_gen_samples, n_fewshot, random_seed, subset=None):
    dataset = TextDataset(
        device,
        dataset_name,
        split,
        n_gen_samples,
        1,
        n_fewshot,
        seed=random_seed
    )

    pos_sequences, neg_sequences = [], []
    pos_labels, neg_labels = [], []

    if subset is None or subset > n_gen_samples:
        subset = n_gen_samples
    for i in range(subset):
        seq, true_label = dataset[i]
        
        if true_label == 1:
            pos_sequences.extend(seq)
            pos_labels.extend(true_label)
        else:
            neg_sequences.extend(seq)
            neg_labels.extend(true_label)
    
    return pos_sequences, neg_sequences, pos_labels, neg_labels


def load_syn_data(syn_data_path, dataset="sst2"):
    syn_data = load_dataset("json", data_files=syn_data_path)["train"]
    # syn_data = []
    # with open(syn_data_path, 'r') as file:
    #     for line in file:
    #         sample = json.loads(line)
    #         syn_data.append(sample)

    syn_pos_sequences = []
    syn_neg_sequences = []

    for example in syn_data:
        sentence = get_first_element(example["inputs"])
        if dataset in ["sst2", "rotten_tomatoes"]:
            sentence = sentence.split("It was")[0].strip()
        elif dataset == "TwitterEmotion":
            sentence = sentence.split("Does the tweet express joy or sadness")[0].strip()
        label = int(example["label"])
        if label == 0:
            syn_neg_sequences.append(sentence)
        else:
            syn_pos_sequences.append(sentence)

    print(f"Pos: {len(syn_pos_sequences)}. Neg: {len(syn_neg_sequences)}")

    return syn_pos_sequences, syn_neg_sequences
    
    
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAP[model_name])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name])
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device


def combine_grad(previous_grads, curr_grads, count):
    new_previous_grads = []
    if count == 0 and previous_grads is None:
        for grad in curr_grads:
            new_previous_grads.append(grad.cuda())
    else:
        for j, grad in enumerate(curr_grads):
            new_previous_grads.append((previous_grads[j] * count + grad.cuda()) / (count + 1))
    
    return new_previous_grads


def remove_grad(previous_grads, curr_grads, count):
    assert count > 0
    new_previous_grads = []
    
    for j, grad in enumerate(curr_grads):
        new_previous_grads.append((previous_grads[j] * count - grad.cuda()) / (count - 1))
    
    return new_previous_grads


def deterministic_usm(list_syn_grads, list_idx, avg_grad, previous_grads=None):
    x = []
    y = copy.deepcopy(list_idx)
    if previous_grads is not None:
        x_grads = copy.deepcopy(previous_grads)
        y_grads = copy.deepcopy(previous_grads)
    else:
        x_grads = None
        y_grads = None
    count_x = 0
    count_y = 0
    
    # Compute the average gradients
    for i in list_idx:
        y_grads = combine_grad(y_grads, list_syn_grads[i], count_y)
        count_y += 1
    
    for idx in tqdm(list_idx):
        # print(f"idx = {idx}")
        syn_grads = list_syn_grads[idx]
        # Compute gain if adding sample to x
        if len(x) == 0:
            x_dist_before = 1
        else:
            x_dist_before = grad_dist(avg_grad, x_grads, metric='cos').item()
        new_x_grads = combine_grad(x_grads, syn_grads, count_x)
        x_dist_after = grad_dist(avg_grad, new_x_grads, metric='cos').item()
        x_dist_diff = x_dist_before - x_dist_after
        # print(f"X: {x_dist_before:.3f} -> {x_dist_after:.3f} ({x_dist_diff:.3f})")
        # Compute gain if removing sample from y
        y_dist_before = grad_dist(avg_grad, y_grads, metric='cos').item()
        new_y_grads = remove_grad(y_grads, syn_grads, count_y)
        y_dist_after = grad_dist(avg_grad, new_y_grads, metric='cos').item()
        y_dist_diff = y_dist_before - y_dist_after
        # print(f"Y: {y_dist_before:.3f} -> {y_dist_after:.3f} ({y_dist_diff:.3f})")
        # Decide the update rule
        if x_dist_diff >= y_dist_diff:
            x.append(idx)
            x_grads = new_x_grads
            count_x += 1
            # print("Add to X")
        else:
            y.remove(idx)
            y_grads = new_y_grads
            count_y -= 1
            # print("Remove from Y")
            
    for x_i, y_i in zip(sorted(x), sorted(y)):
        assert x_i == y_i, f"{x_i} != {y_i}"
    
    return x


def filtering(args, training_dir, model, tokenizer, device, skip_samples=0, num_out=None):
  for run_id, run in tqdm(enumerate(training_dir)):
    file_path = os.path.join(run, f'{args.json_file}.jsonl')
    if not os.path.exists(file_path) or run_id < skip_samples:
        print(file_path)
        continue

    if args.filter_method == "first":
        filtered_samples = extract_first_samples_per_label(
            file_path,
            top_n=args.top_n,
            per_label=args.per_label,
            interleave=args.interleave_label
        )
        new_json_file = f'first_{args.top_n}samples'
        if args.per_label:
            new_json_file += '_per_label'
        if args.interleave_label:
            new_json_file += '_interleave'
    elif args.filter_method == "top_score":
        filtered_samples = extract_top_samples_per_label(
            file_path,
            args.coeff_perplexity,
            top_n=args.top_n,
            per_label=args.per_label,
            interleave=args.interleave_label,
            balance_score=args.balance_score
        )
        new_json_file = args.json_file + f'_top{args.top_n}_score_alpha{args.coeff_perplexity}'
        if args.per_label:
            new_json_file += '_per_label'
        if args.interleave_label:
            new_json_file += '_interleave'
        if args.balance_score:
            new_json_file += '_balance_score'
    elif args.filter_method == "bottom_score":
        filtered_samples = extract_top_samples_per_label(
            file_path,
            args.coeff_perplexity,
            top_n=args.top_n,
            per_label=args.per_label,
            interleave=args.interleave_label,
            reverse=True,
            balance_score=args.balance_score
        )
        new_json_file = args.json_file + f'_bottom{args.top_n}_score_alpha{args.coeff_perplexity}'
        if args.per_label:
            new_json_file += '_per_label'
        if args.interleave_label:
            new_json_file += '_interleave'
        if args.balance_score:
            new_json_file += '_balance_score'
    elif args.filter_method == "greedy_selection":
        filtered_samples = greedy_selection(
            file_path,
            args, 
            model,
            tokenizer,
            device,
        )
        new_json_file = args.json_file + f'_greedy_top{args.top_n}'
        if args.interleave_label:
            new_json_file += '_interleave'
    else:
        filtered_samples = filter_synthetic_data(
            args,
            file_path,
            model,
            tokenizer
        )
        new_json_file = get_output_file_name(args)

    # Save the top samples to a new file (optional)
    output_file = str(file_path).replace(f'{args.json_file}.jsonl', f'{new_json_file}.jsonl')
    # print(output_file)

    output_to_jsonl(
        args, 
        filtered_samples, 
        output_file, 
        post_processing=args.clean, 
        num_out=num_out
    )

    
def compute_average_grads(args, model, tokenizer, sequences, labels):
    """Compute average gradients.

    Args:
        args: Arguments
        model: Model
        tokenizer: Tokenizer
        sequences: List of samples
        labels: Labels

    Returns:
        average_grads: Average gradients
    """
    lm_embeddings = model.get_input_embeddings()
    num_samples = len(sequences)
    text_labels = []
    prompt_lengths = []  # Collect prompt lengths for dataset with two prompts
    if args.dataset in ["sst2", "rotten_tomatoes"]:
        sequences = [seq + " It was " for seq in sequences]
        for seq in sequences:
            prompt_len = len(
                tokenizer(seq)["input_ids"]
            )  # Total token count for sst2 prompt + sequence
            prompt_lengths.append(prompt_len)
        for label in labels:
            text_labels.append("bad" if label.flatten() == 0 else "great")
    elif args.dataset == "TwitterEmotion":
        sequences = [
            seq + " Does the tweet express joy or sadness?\n" for seq in sequences
        ]
        for seq in sequences:
            prompt_len = len(
                tokenizer(seq)["input_ids"]
            )  # Total token count for TwitterEmotion prompt + sequence
            prompt_lengths.append(prompt_len)
        for label in labels:
            text_labels.append("sadness" if label.flatten() == 0 else "joy")
  
    average_grads = None

    for i in tqdm(range(num_samples)):
        seq = sequences[i]
        text_label = text_labels[i]
        orig_batch = tokenizer(
            seq, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        label = tokenizer(
            text_label, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        label = label["input_ids"].view(-1)
        true_embeds = lm_embeddings(orig_batch["input_ids"])
        curr_grads = compute_grads_lm(
            model, true_embeds, orig_batch["attention_mask"], label
        )

        if average_grads is None:
            average_grads = []
            for grad in curr_grads:
                average_grads.append(grad.detach() / num_samples)
        else:
            for j, grad in enumerate(curr_grads):
                average_grads[j].add_(grad.detach() / num_samples)
        del curr_grads
        torch.cuda.empty_cache()

    return average_grads


def calculate_recon_loss_ids(list_sequence, list_label, avg_grad, model, tokenizer, dataset="sst2"):
    lm_embeddings = model.get_input_embeddings()
    list_loss = []
    
    for seq, text_label in zip(list_sequence, list_label):
        if dataset in ["sst2", "rotten_tomatoes"]:
            orig_batch = tokenizer(
                seq + " It was ", padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
        elif dataset == "TwitterEmotion":
            orig_batch = tokenizer(
                seq + " Does the tweet express joy or sadness?\n", padding=True, truncation=True, return_tensors="pt"
            ).to(model.device)
        label = tokenizer(
            text_label, padding=True, truncation=True, return_tensors="pt"
        ).to(model.device)
        label = label["input_ids"].view(-1)
        true_embeds = lm_embeddings(orig_batch["input_ids"])
        curr_grads = compute_grads_lm(
            model, true_embeds, orig_batch["attention_mask"], label
        )
        list_loss.append(grad_dist(
            avg_grad, 
            curr_grads, 
            metric='cos'
        ).item())
        del curr_grads
        torch.cuda.empty_cache()
    
    return list_loss


if __name__ == '__main__':
    # Load parameters
    args = get_args()
    print(args)

    # Load model
    tokenizer, model, device = load_model(args.model_name)
    
    file_dir = args.file_dir
    training_dir = glob.glob(os.path.join(file_dir, '*'))
    print(len(training_dir))
    
    filtering(
        args,
        training_dir,
        model,
        tokenizer,
        device,
        num_out=5 if args.top_n == 3 else None
    )