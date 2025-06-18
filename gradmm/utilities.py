import os
import re
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_all_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_rng_states(work_dir):
    """Save the RNG states of PyTorch, NumPy, and Python's random module to a file.

    Args:
        filepath (str): The file path to save the RNG states.
    """
    rng_states = {
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(rng_states, os.path.join(work_dir, "rng_states.pth"))
    print(f"RNG states saved to {os.path.join(work_dir, 'rng_states.pth')}")


def load_rng_states(work_dir):
    """Load the RNG states of PyTorch, NumPy, and Python's random module from a file.

    Args:
        filepath (str): The file path to load the RNG states from.
    """
    rng_states = torch.load(
        os.path.join(work_dir, "rng_states.pth"), weights_only=False
    )
    torch.random.set_rng_state(rng_states["torch_rng_state"])
    np.random.set_state(rng_states["numpy_rng_state"])
    random.setstate(rng_states["python_rng_state"])
    print(f"RNG states loaded from {os.path.join(work_dir, 'rng_states.pth')}")


def count_lines(file_path: str):
    with open(file_path, "r") as file:
        line_count = sum(1 for _ in file)
    
    return line_count


def get_args_flags(args):
    """Get experiment flags.

    Args:
        args: Arguments

    Returns:
        flags: Experiment flags
    """
    flags = f"{args.dataset.upper()}-{args.split}-{args.model_name}"
    flags += f"-nreal{args.n_gen_samples}"
    flags += f"-steps{args.n_steps}"
    flags += f"-nsyn{args.n_gen}"
    flags += f"-{args.opt_alg}"
    if args.use_dp:
        flags += f"-dp_eps{args.dp_epsilon}"
    flags += f"-rho{args.admm_rho}"
    flags += f"-inner{args.admm_inner_steps}"
    flags += f"-seed{args.rng_seed}"
    
    return flags


def compute_grads_lm(
    model,
    x_embeds,
    attention_mask,
    y_labels,
    create_graph=False,
    gen_grad_clip="",
):
    criterion = nn.CrossEntropyLoss()
    outputs = model(inputs_embeds=x_embeds, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]

    # Short text labels, such as "Yes" or "No", can sometimes span more than one
    # token. For classification purposes, we can use only the first token. In
    # MeZO, this is addressed by introducing the "option length" field. For now,
    # we only need to ensure that the pos and neg labels are distinct.
    if y_labels.shape[0] > 1:
        y_labels = y_labels[:1]

    if x_embeds.shape[0] > 1:
        # x_embeds is more than one
        y_labels = y_labels.repeat(x_embeds.shape[0])

    loss = criterion(logits, y_labels)
    grads = torch.autograd.grad(
        loss,
        (param for param in model.parameters() if param.requires_grad),
        create_graph=create_graph,
        allow_unused=True,
    )
    if gen_grad_clip == "elem":
        grads = [g.clamp_(min=-1, max=1) if g is not None else None for g in grads]
    elif gen_grad_clip == "norm":
        norm = torch.sqrt(sum((g**2).sum() for g in grads if g is not None))
        if norm > 1:
            grads = [g.div_(norm) if g is not None else None for g in grads]
    
    return grads


def compute_grads_lm_ids(
    model,
    ids,
    attention_mask,
    y_labels,
    create_graph=False,
    gen_grad_clip="",
):
    criterion = nn.CrossEntropyLoss()
    outputs = model(input_ids=ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]
    # Short text labels, such as "Yes" or "No", can sometimes span more than one
    # token. For classification purposes, we can use only the first token. In
    # MeZO, this is addressed by introducing the "option length" field. For now,
    # we only need to ensure that the pos and neg labels are distinct.
    if y_labels.shape[0] > 1:
        y_labels = y_labels[:1]
    if ids.shape[0] > 1:
        y_labels = y_labels.repeat(ids.shape[0])
    loss = criterion(logits, y_labels)
    grad = torch.autograd.grad(
        loss,
        (param for param in model.parameters() if param.requires_grad),
        create_graph=create_graph,
        allow_unused=True,
    )
    if gen_grad_clip == "elem":
        grad = [g.clamp_(min=-1, max=1) if g is not None else None for g in grad]
    elif gen_grad_clip == "norm":
        norm = torch.sqrt(sum((g**2).sum() for g in grad if g is not None))
        if norm > 1:
            grad = [g.div_(norm) if g is not None else None for g in grad]
    
    return grad


def cos_sim(x, y):
    return (x * y).sum() / (x.norm(p=2) * y.norm(p=2))


def cos_sim_batch(x, y):
    """Compute batch-wise cosine similarity between two batches of vectors.

    Args:
        x: Tensor of shape (batch_size, vector_dim)
        y: Tensor of shape (batch_size, vector_dim)

    Returns:
        Tensor of shape (batch_size,) containing cosine similarity for each pair.
    """
    # Ensure x and y are normalized to prevent division by zero
    x_norm = x / x.norm(p=2, dim=1, keepdim=True)
    y_norm = y / y.norm(p=2, dim=1, keepdim=True)

    # Compute cosine similarity for each vector in the batch
    cos_similarities = (x_norm * y_norm).sum(dim=1)

    return cos_similarities.mean()


def grad_dist(target_grads, curr_grads, args, previous_grad=None):
    """Calculate gradient distance.

    Args:
        target_grads: target gradients
        curr_grads: current gradients
        args: additional arguments
        previous_grad: previous gradients

    Returns:
        ret: objective
    """
    ret = 0.0
    n_g = 0
    if previous_grad is None:
        previous_grad = [None for _ in range(len(curr_grads))]
    for g1, g2, g3 in zip(target_grads, curr_grads, previous_grad):
        if (g1 is not None) and (g2 is not None):
            if g3 is None:
                g3 = torch.zeros_like(g2)
            else:
                g3 = g3.to(g2.device)  # pytype: disable=attribute-error
        if args.loss == "cos":
            ret += 1 - cos_sim(g1, g2 + g3)
        elif args.loss == "dlg":
            ret += (g1 - g2 - g3).square().sum()
        elif args.loss == "tag":
            ret += (g1 - g2 - g3).square().sum() + args.tag_factor * torch.abs(
                g1 - g2 - g3
            ).sum()
        else:
            assert False
        n_g += 1
    if args.loss == "cos":
        ret /= n_g
    
    return ret


def get_closest_tokens(
    inputs_embeds, unused_tokens, embeddings_weight, metric="cos"
):
    """Get closest tokens.

    Args:
        inputs_embeds: Input embeddings, shape (bs, seq_len, embed_dim)
        unused_tokens: List of unused tokens
        embeddings_weight: Embeddings weight, shape (vocab_size, embed_dim)
        metric: Metric to use

    Returns:
        d: Distance matrix
        cos_ids: Closest token ids, shape (seq_len)
    """
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    if metric == "l2":
        d = torch.cdist(inputs_embeds, embeddings_weight, p=2)
    elif metric == "cos":
        dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
        norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
        norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
        d = -dp / (norm1 * norm2)
    else:
        assert False
    d[:, :, unused_tokens] = 1e9
    
    return d, d.min(dim=2)[1]


def get_prefix(
    args, fewshot_seqs, fewshot_labels, default_prefix="The", label_text="great"
):
    """Get prefix for generation.

    Args:
        args: Arguments
        fewshot_seqs: Fewshot sequences
        fewshot_labels: Fewshot labels
        default_prefix: Default prefix

    Returns:
        prefix: Prefix
    """
    if args.dataset in [
        "sst2",
        "rotten_tomatoes",
        "imdb",
        "rtpolarity",
    ]:
        prompt_text = " It was "
        label_mapping = {0: "bad", 1: "great"}
    elif args.dataset == "TwitterEmotion":
        prompt_text = " Does the tweet express joy or sadness?\n"
        label_mapping = {0: "sadness", 1: "joy"}
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    prefix = ""
    for seq, label in zip(fewshot_seqs, fewshot_labels):
        if isinstance(prompt_text, list):
            prefix += (
                seq[0]
                + prompt_text[0]
                + seq[1]
                + prompt_text[1]
                + label_mapping[label.item()]
                + "\n\n"
            )
        else:
            prefix += seq + prompt_text + label_mapping[label.item()] + "\n\n"
    
    if prefix:
        return [prefix]
    else:
        if args.dataset in [
            "rotten_tomatoes",
            "rtpolarity",
        ]:
            default_prefix = (
                "You are now a movie critic. You are provided with a sentiment label"
                " (either 'great' or 'bad'). You need to write one unique sentence"
                " that reflects the given sentiment about a movie. Your writing style"
                " should be consistent with typical movie reviews. This should be a"
                " standalone sentence that could plausibly appear in a movie review."
                " Ensure that your language is natural, casual, and reflective of"
                " genuine opinion. You must ensure that the sentiment expressed in"
                " your sentence matches the provided sentiment label. Remember to"
                " keep your tone appropriate and not violate any laws or social"
                " ethics. Please be creative and write only one sentence. The"
                f" sentiment of the movie review is: {label_text}\n."
                " Answer:"
            )
        elif args.dataset == "imdb":
            default_prefix = (
                "You are now a movie critic. You are provided with a sentiment label"
                " (either 'great' or 'bad'). You need to write a film review that"
                " reflects the given sentiment about a movie. Your writing style"
                " should be consistent with typical movie reviews. This should be a "
                " highly polar review for a movie that could plausibly be posted on"
                " IMDB. Ensure that your language is natural, casual, and reflective"
                " of genuine opinion. You must ensure that the sentiment expressed"
                " in your review matches the provided sentiment label. Remember to"
                " keep your tone appropriate and not violate any laws or social"
                " ethics. Please be creative and write an unique movie review. The"
                f" sentiment of the movie review is: {label_text}\n. Answer:"
            )
        elif args.dataset == "TwitterEmotion":
            default_prefix = (
                "You are now a person using twitter. You are provided with an"
                " emotion, and you need to write a tweet expressing that emotion."
                " Your writing style must be consistent with the tweets on twitter."
                " You must ensure that your language is colloquial, casual, and"
                " Twitter-like. You are given a length requirement. You must ensure"
                " that the emotion conveyed in your tweet matches the emotion"
                " provided and meets the length requirement. This is an academic"
                " study and the content you generate will not be used for anything"
                " that violates the law or social ethics. Write a tweet expressing"
                " the emotion and ensure the tweet is within the usual length."
                " Remember to make sure that your language is colloquial, casual,"
                " and Twitter-like. Please be creative and write only one unique"
                f" tweets. The emotion of twitter is: {label_text}\n."
                " Answer:"
            )

    return [default_prefix]


def get_topk_closest_tokens(
    inputs_embeds,
    unused_tokens,
    embeddings_weight,
    model,
    tokenizer,
    prefix,
    prompt_len,
    include_prefix=False,
    topk=1000,
):
    """Get closest tokens in top-k decoding.

    Args:
        inputs_embeds: Input embeddings, shape (bs, seq_len, embed_dim)
        unused_tokens: List of unused tokens
        embeddings_weight: Embeddings weight, shape (vocab_size, embed_dim)
        model: Model
        tokenizer: Tokenizer
        prefix: Prefix
        prompt_len: Prompt length
        include_prefix: Whether to include prefix
        topk: Number of tokens to consider

    Returns:
        d: Distance matrix
        cos_ids: Closest token ids, shape (seq_len)
    """
    embeddings_weight = embeddings_weight.repeat(inputs_embeds.shape[0], 1, 1)
    dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
    norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
    norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
    d = -dp / (norm1 * norm2)
    d[:, :, unused_tokens] = 1e9
    d = torch.argsort(d, dim=2)
    gen_max_tokens = inputs_embeds.shape[1]
    prefix_ids = tokenizer(
        prefix, padding=True, truncation=True, return_tensors="pt"
    )["input_ids"][0].tolist()
    gen_bs = inputs_embeds.shape[0]
    mapped_token_list = [prefix_ids.copy() for _ in range(gen_bs)]
    # If the prefix is too long, which means we use ICL, do not include the prefix
    if len(mapped_token_list) > 5:
        include_prefix = False
        start_token_index = 0
    elif include_prefix:
        start_token_index = len(mapped_token_list)
    else:
        start_token_index = 0

    for i in range(start_token_index, gen_max_tokens):
        # Don't need to decode for the prompt
        if isinstance(prompt_len, list):
            if i + prompt_len[1] >= gen_max_tokens:
                # mapped_token_list.append(0)
                for j in range(len(mapped_token_list)):
                    mapped_token_list[j].append(0)
                continue
        else:
            if i + prompt_len >= gen_max_tokens:
                # mapped_token_list.append(0)
                for j in range(len(mapped_token_list)):
                    mapped_token_list[j].append(0)
                    # mapped_token_list.append([0 for _ in range(gen_bs)])
                continue
        curr_input = {
            "input_ids": torch.asarray(mapped_token_list, device=d.device),
        }
        outputs = model.generate(
            **curr_input,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        score = outputs.scores[0]  # shape: [gen_bs, vocab_size]
        gen_bs, vocab_size = score.shape

        score_argsort = torch.argsort(score, dim=1, descending=True)

        for row in range(gen_bs):
            sorted_tokens = score_argsort[row]  # shape: [vocab_size]

            row_topk = d[row, i, :topk]  # shape: [topk]

            presence_mask = torch.isin(sorted_tokens, row_topk)  # shape: [vocab_size]

            matches = torch.nonzero(presence_mask, as_tuple=True)[0]
            if matches.numel() > 0:
                first_match_idx = matches[0].item()
                chosen_token = sorted_tokens[first_match_idx].item()
                mapped_token_list[row].append(chosen_token)

    if not include_prefix:
        for i in range(len(mapped_token_list)):
            mapped_token_list[i] = mapped_token_list[i][len(prefix_ids) :]
            # mapped_token_list = mapped_token_list[-gen_max_tokens:]

    return d, torch.asarray(mapped_token_list, device=d.device)


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, filter_value=-float("Inf")
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (vocabulary size) top_k > 0: keep only top
        k tokens with highest probability (top-k filtering). top_p > 0.0: keep the
        top tokens with cumulative probability >= top_p (nucleus filtering).
        Nucleus filtering is described in Holtzman et al.
        (http://arxiv.org/abs/1904.09751)

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(
    inputs_embeds,
    unused_tokens,
    embeddings_weight,
    model,
    tokenizer,
    prefix,
    prompt_len,
    include_prefix=False,
    temperature=1,
    top_k=0,
    top_p=0.0,
):
    dp = torch.bmm(inputs_embeds, embeddings_weight.transpose(1, 2))
    norm1 = inputs_embeds.norm(p=2, dim=2).unsqueeze(2)
    norm2 = embeddings_weight.norm(p=2, dim=2).unsqueeze(1)
    d = -dp / (norm1 * norm2)
    d[:, :, unused_tokens] = -1e9
    prob_d = F.softmax(d, dim=-1)
    gen_max_tokens = inputs_embeds.shape[1]
    prefix_ids = tokenizer(
        prefix, padding=True, truncation=True, return_tensors="pt"
    )["input_ids"][0].tolist()
    mapped_token_list = prefix_ids
  # If the prefix is too long, which means we use ICL, do not include the prefix
    if len(mapped_token_list) > 5:
        include_prefix = False
        start_token_index = 0
    elif include_prefix:
        start_token_index = len(mapped_token_list)
    else:
        start_token_index = 0
    
    with torch.no_grad():
        for i in range(start_token_index, gen_max_tokens):
            # Don't need to decode for the prompt
            if isinstance(prompt_len, list):
                if i + prompt_len[1] >= gen_max_tokens:
                    mapped_token_list.append(0)
                    continue
            else:
                if i + prompt_len >= gen_max_tokens:
                    mapped_token_list.append(0)
                    continue
            curr_input = {
                "input_ids": torch.asarray([mapped_token_list], device=d.device),
            }
            outputs = model.generate(
                **curr_input,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            next_token_logits = outputs.scores[0][-1, :] / temperature
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                (F.softmax(filtered_logits, dim=-1) + prob_d[0, i, :].squeeze(0)) / 2,
                num_samples=1,
            )
            mapped_token_list.append(next_token.item())
    
    if not include_prefix:
        mapped_token_list = mapped_token_list[-gen_max_tokens:]
    
    return d, torch.tensor(mapped_token_list, device=d.device)


def get_perplexity_loss(x_embeds, label_ids, model):
    output = model(inputs_embeds=x_embeds)
    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # shift_labels[:, :-16] = -100
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    
    return loss


def get_reconstruction_loss(
    model,
    x_embeds,
    attention_mask,
    y_labels,
    true_grads,
    args,
    create_graph=False,
    previous_grad=None,
    return_grads=False,
):
    """Calculate reconstruction loss from embeddings.

    Args:
        model: Model
        x_embeds: Embeddings
        attention_mask: Attention mask
        y_labels: True labels
        true_grads: True gradients
        args: Arguments
        create_graph: Whether to create graph
        previous_grad: Previous gradients
        return_grads: Whether to return gradients

    Returns:
        grad_dist: Gradient distance
        return_grads (optional): Gradients
    """
    grads = compute_grads_lm(
        model,
        x_embeds,
        attention_mask,
        y_labels,
        create_graph=create_graph,
        gen_grad_clip=args.gen_grad_clip,
    )
    if not return_grads:
        return grad_dist(true_grads, grads, args, previous_grad=previous_grad)
    
    return_grads = []
    for grad in grads:
        if grad is not None:
            return_grads.append(grad.detach())
        else:
            return_grads.append(None)

    return (
        grad_dist(true_grads, grads, args, previous_grad=previous_grad),
        return_grads,
    )


def get_reconstruction_loss_ids(
    model,
    ids,
    attention_mask,
    y_labels,
    true_grads,
    args,
    create_graph=False,
    previous_grad=None,
    return_grads=False,
):
    """Calculate reconstruction loss from token ids.

    Args:
        model: Model
        ids: Token ids
        attention_mask: Attention mask
        y_labels: True labels
        true_grads: True gradients
        args: Arguments
        create_graph: Whether to create graph
        previous_grad: Previous gradients
        return_grads: Whether to return gradients

    Returns:
        grad_dist: Gradient distance
        return_grads (optional): Gradients
    """
    grads = compute_grads_lm_ids(
        model,
        ids,
        attention_mask,
        y_labels,
        create_graph=create_graph,
        gen_grad_clip=args.grad_clip,
    )
    if not return_grads:
        return grad_dist(true_grads, grads, args, previous_grad=previous_grad)
    
    return_grads = []
    for grad in grads:
        if grad is not None:
            return_grads.append(grad.detach())
        else:
            return_grads.append(None)

    return (
        grad_dist(true_grads, grads, args, previous_grad=previous_grad),
        return_grads,
    )


def get_embed_diff(args, model, ids, avg_embeds):
    """Calculate embedding difference.

    Args:
        args: Arguments
        model: Model
        ids: Token ids, shape (bs, seq_len)
        avg_embeds: Average embeddings, shape (1, seq_len, embed_dim)

    Returns:
        embed_diff: Embedding difference
    """
    lm_embeddings_weight = model.get_input_embeddings().weight
    x_embeds = lm_embeddings_weight[ids]
    
    if args.embed_loss == "cos":
        embed_diff = 1 - cos_sim(x_embeds.mean(dim=1), avg_embeds)
    elif args.embed_loss == "dlg":
        embed_diff = (x_embeds.mean(dim=1) - avg_embeds).square().sum()
    else:
        embed_diff = torch.tensor(0.0, device=avg_embeds.device)
    
    return embed_diff


def remove_padding(tokenizer, ids, first_prompt_end_index):
    if len(ids) > first_prompt_end_index:
        return [
            tokenizer.decode(ids[:first_prompt_end_index]),
            tokenizer.decode(ids[first_prompt_end_index:]),
        ]
    else:
        return tokenizer.decode(ids)


def extract_unique_words(sentences):
    """Extracts unique words from a list of sentences, removing punctuation and numbers.

    Args:
        sentences: A list of sentences.

    Returns:
        A set of unique words.
    """
    all_words = []
    for sentence in sentences:
        # Remove punctuation and numbers using regular expressions
        words = re.sub(r"[^A-Za-z\s]", "", sentence).split()
        all_words.extend(words)

    unique_words = set(all_words)
    
    return unique_words


def extract_first_words(sentences, min_length=3):
    """Extracts the first word from each sentence in a list.

    Includes only pronouns or words with at least min_length characters.

    Args:
        sentences (list): A list of sentences (strings).

    Returns:
        list: A list of the filtered first words from the sentences.
    """
    # List of common pronouns to include
    pronouns = {"I", "you", "he", "she", "it", "we", "they"}

    # Process sentences
    first_words = [
        word
        for sentence in sentences
        if (words := sentence.split())  # Ensure sentence is not empty
        and (word := words[0])  # Extract the first word
        and (
            len(word) >= min_length or word.lower() in pronouns
        )  # Filter condition
    ]

    return first_words


def construct_fewshot_prompt(
    pos_seqs, pos_labels, neg_seqs, neg_labels, first_name, second_name
):
    """Construct fewshot prompt."""
    # Ensure the lists have the same length
    assert len(pos_seqs) == len(pos_labels)
    assert len(neg_seqs) == len(neg_labels)

    # Interleave positive and negative examples
    few_shot_prompt = []
    for pos_seq, pos_label, neg_seq, neg_label in zip(
        pos_seqs, pos_labels, neg_seqs, neg_labels
    ):
        few_shot_prompt.append(
            f"{first_name}: {pos_seq}\n{second_name}: {pos_label}\n"
        )
        few_shot_prompt.append(
            f"{first_name}: {neg_seq}\n{second_name}: {neg_label}\n"
        )

    # Join the interleaved examples into a single string for the prompt
    few_shot_prompt_str = "".join(few_shot_prompt)

    return few_shot_prompt_str
