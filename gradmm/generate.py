import os
import json
import pickle
import random
import shutil
import sys
import time
import math

import numpy as np
import regex as re
import wandb
import torch
from torch import optim
from transformers import AutoModelForCausalLM, AutoTokenizer

from args_factory import get_args
from data_utils import BatchDatasetLoader, TextDataset
from init import get_init_lm
from utilities import (
    compute_grads_lm,
    cos_sim,
    count_lines,
    extract_first_words,
    get_args_flags,
    get_closest_tokens,
    get_embed_diff,
    get_perplexity_loss,
    get_prefix,
    get_reconstruction_loss,
    get_reconstruction_loss_ids,
    get_topk_closest_tokens,
    grad_dist,
    load_rng_states,
    save_rng_states,
    remove_padding,
    sample_sequence,
    set_all_seeds,
)


unused_tokens = None


def get_loss(
    args,
    model,
    ids,
    x_embeds,
    attention_mask,
    true_labels,
    true_grads,
    avg_embeds,
    create_graph=False,
    previous_grad=None,
    return_grads=False,
):
    """Get all losses.

    Args:
        args: Arguments
        model: Model
        ids: Token ids, shape (bs, seq_len)
        x_embeds: Embeddings, shape (bs, seq_len, embed_dim)
        attention_mask: Attention mask, shape (bs, seq_len)
        true_labels: True labels, shape (bs, 1)
        true_grads: True gradients, list
        avg_embeds: Average embeddings, shape (1, seq_len, embed_dim)
        create_graph: Whether to create graph
        previous_grad: Previous gradients
        return_grads: Whether to return gradients

    Returns:
        return_dict: Dictionary of losses
    """
    perplexity = model(
        input_ids=ids, attention_mask=attention_mask, labels=ids
    ).loss
    rec_loss_embeds = get_reconstruction_loss(
        model,
        x_embeds,
        attention_mask,
        true_labels,
        true_grads,
        args,
        create_graph=create_graph,
        previous_grad=previous_grad,
    )
    return_dict = {
        "perplexity": perplexity,
        "rec_loss_embeds": rec_loss_embeds,
    }
    rec_loss_ids = get_reconstruction_loss_ids(
        model,
        ids,
        attention_mask,
        true_labels,
        true_grads,
        args,
        create_graph=create_graph,
        previous_grad=previous_grad,
        return_grads=return_grads,
    )
    return_dict["embed_diff_ids"] = get_embed_diff(args, model, ids, avg_embeds)
    
    if return_grads:
        rec_loss_ids, new_grad = rec_loss_ids[0], rec_loss_ids[1]
        return_dict["new_grad"] = new_grad
    return_dict["rec_loss_ids"] = rec_loss_ids
    return_dict["tot_loss"] = rec_loss_ids + args.coeff_perplexity * perplexity

    return return_dict


def generation(
    args,
    device,
    metric,
    true_grads,
    true_labels,
    init_embeds,
    init_prompt_length,
    avg_embeds,
    few_shot_seqs,
    few_shot_labels,
    tokenizer,
    model,
    token_candidates,
    list_prefix,
    only_init=False,
    previous_grad=None,
):
    """Generate synthetic data.

    Args:
        args: Arguments
        device: Device
        metric: Metric
        true_grads: True gradients, list
        true_labels: True labels, shape (bs, seq_len)
        init_embeds: Init embeddings, shape (bs, seq_len, emb_dim)
        init_prompt_length: Init prompt length
        avg_embeds: Average true embeddings over sequence length, shape (bs,
        emb_dim)
        few_shot_seqs: Few shot sequences
        few_shot_labels: Few shot labels
        tokenizer: Tokenizer
        model: Model
        token_candidates: List of token candidates
        list_prefix: List of prefix
        only_init: Take the best init embedding. Don't run optimization
        previous_grad: Previous gradients

    Returns:
        res: Dictionary of results
    """
    print(f"Metric: {metric}")
    lm_embeddings = model.get_input_embeddings()
    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)
    init_embeds = init_embeds.to(device)

    global unused_tokens
    if unused_tokens is None:
        unused_tokens = []
        # model embedding size is larger than the vocab size
        # the final tokens are never used
        for i in range(tokenizer.vocab_size, lm_embeddings.weight.shape[0]):
            unused_tokens.append(i)
        unused_tokens.append(tokenizer.pad_token_id)
        unused_tokens.append(tokenizer.eos_token_id)
        # Remove tokens that contain special characters
        for num in range(len(tokenizer)):
            text = tokenizer.decode(num)
            for symbol in ["\n", '"', "#", "..."]:
                if symbol in text:
                    unused_tokens.append(num)
        if args.drop_non_english_tokens:
            print("Dropping non-english tokens")
            non_english_tokens = []
            english_tokens = []
            # pattern = r"[A-Za-z\p{P}\s]+"
            pattern = r"^[a-z]+(?:[\'-][a-z]+)*$"

            for num in range(tokenizer.vocab_size):
                text = tokenizer.decode(num)
                text = text.lower().strip()
                text = re.sub(r"^\W+|\W+$", "", text)
                if text == "":
                    continue
                if not re.fullmatch(pattern, text) or len(text) > 12:
                    non_english_tokens.append(num)
                else:
                    english_tokens.append(num)
            with open(os.path.join(args.work_dir, "non_english_tokens.json"), "w") as f:
                json_to_write = {}
                for t in non_english_tokens:
                    json_to_write[t] = tokenizer.decode(t)
                f.write(json.dumps(json_to_write, indent=2))
            with open(os.path.join(args.work_dir, "english_tokens.json"), "w") as f:
                json_to_write = {}
                for t in english_tokens:
                    json_to_write[t] = tokenizer.decode(t)
                f.write(json.dumps(json_to_write, indent=2))
            unused_tokens.extend(non_english_tokens)

        if args.use_sample_tokens_only:
            not_sample_tokens = set(range(tokenizer.vocab_size)) - set(token_candidates)
            for i in not_sample_tokens:
                unused_tokens.append(i)
        if token_candidates is not None:
            print(f"Number of used tokens: {len(token_candidates)}")
        unused_tokens = list(set(unused_tokens))
        
        if args.drop_change_line_characters:
            print("Dropping change line characters")
            unused_tokens = []
            pattern = r"^[a-z]+(?:[\'-][a-z]+)*$"

            for token in range(tokenizer.vocab_size):
                text = tokenizer.decode(token)
                # Check for special characters or length constraint
                if (
                    any(char in text for char in "*;:_\n-'<>:{}[]()/\\|=+%@~`^#$&")
                    or len(text) > 17
                ):
                    unused_tokens.append(token)
                    continue
                # Normalize text and check against pattern
                text_lower = text.lower().strip()
                text_lower = re.sub(r"^\W+|\W+$", "", text_lower)
                if not re.fullmatch(pattern, text_lower):
                    unused_tokens.append(token)
  
    print(f"Number of unused tokens: {len(unused_tokens)}")

    if args.dataset in [
        "sst2",
        "rotten_tomatoes",
        "imdb",
        "rtpolarity",
    ]:
        prompt_text = " It was "
        prompt = tokenizer(
            prompt_text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        prompt_ids = prompt["input_ids"].view(-1)
        prompt_len = prompt_ids.shape[0]
        prompt_embeddings = lm_embeddings(prompt_ids)
        text_labels = []
        for label in true_labels:
            text_labels.append("bad" if label.flatten() == 0 else "great")
        true_labels_tokenized = tokenizer(
            text_labels, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        true_labels_tokenized = true_labels_tokenized["input_ids"].view(-1)
    elif args.dataset == "TwitterEmotion":
        prompt_text = " Does the tweet express joy or sadness?\n"
        prompt = tokenizer(
            prompt_text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        prompt_ids = prompt["input_ids"].view(-1)
        prompt_len = prompt_ids.shape[0]
        prompt_embeddings = lm_embeddings(prompt_ids)
        text_labels = []
        for label in true_labels:
            text_labels.append("sadness" if label.flatten() == 0 else "joy")
        true_labels_tokenized = tokenizer(
            text_labels, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        true_labels_tokenized = true_labels_tokenized["input_ids"].view(-1)
    else:
        raise ValueError("Unsupported dataset: %s" % args.dataset)

    # Get initial embeddings + set up opt
    gen_shape = (
        args.gen_bs,
        args.gen_max_tokens,
        lm_embeddings.weight.shape[-1],
    )

    if "real" in args.init:
        print(f"Using real embeddings with shape {init_embeds.shape}.")
        # Note that init_embeds already has the prompt embeddings.
        x_embeds = init_embeds.clone()
        x_embeds.requires_grad_(True)
        attention_mask = torch.ones(
            x_embeds.shape[0], x_embeds.shape[1], device=device
        ).long()
        args.first_prompt_end_index = init_prompt_length
    else:
        print("Generating random initial embeddings.")
        attention_mask = torch.ones(
            gen_shape[0], gen_shape[1], device=device
        ).long()
        x_embeds = get_init_lm(
            args,
            model,
            unused_tokens,
            gen_shape,
            prompt_embeddings,
            attention_mask,
            true_labels_tokenized,
            true_grads,
            lm_embeddings_weight,
            tokenizer,
            previous_grad=previous_grad,
        )
        args.first_prompt_end_index = args.gen_max_tokens

    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)

    # for admm
    z_embeds = torch.zeros_like(x_embeds)
    lambda_embeds = torch.zeros_like(x_embeds)
    if args.opt_alg == "adam":
        opt = optim.Adam([x_embeds], lr=args.lr)
    elif args.opt_alg == "sgd" or args.opt_alg == "admm_sgd":
        opt = optim.SGD([x_embeds], lr=args.lr, momentum=0.9)
    elif args.opt_alg == "bfgs":
        opt = optim.LBFGS([x_embeds], lr=args.lr)
    elif args.opt_alg == "bert-adam":
        opt = torch.optim.AdamW(
            [x_embeds], lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
        )
    else:
        opt = optim.Adam([x_embeds], lr=args.lr)

    if args.lr_decay_type == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            opt, step_size=50, gamma=args.lr_decay
        )
    elif args.lr_decay_type == "LambdaLR":

        def lr_lambda(current_step: int):
            return max(
                0.0,
                float(args.lr_max_it - current_step) / float(max(1, args.lr_max_it)),
            )

        lr_scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError("Unsupported lr_decay_type: %s" % args.lr_decay_type)
    print("Nsteps:", args.n_steps, flush=True)

    # Main loop
    best_final_error, best_final_x = None, x_embeds.detach().clone()
    best_norm_diff, best_embed_diff = None, None
    best_rec_loss, best_reg_loss = None, None

    # Check if we need to run optimization
    if only_init:
        prev_n_steps = args.n_steps
        args.n_steps = 0

    for it in range(args.n_steps):
        if isinstance(prompt_embeddings, list):
        # First prompt
            x_embeds.data[
                :,
                args.first_prompt_end_index
                - prompt_len[0] : args.first_prompt_end_index,
                :,
            ] = (
                prompt_embeddings[0].detach().clone()
            )
            # Second prompt
            x_embeds.data[:, -prompt_len[1] :, :] = (
                prompt_embeddings[1].detach().clone()
            )
        else:
            x_embeds.data[:, -prompt_len:, :] = prompt_embeddings.detach().clone()
        
        t_start = time.time()
        if args.opt_alg == "admm":
            # x + rho^-1 * lambda
            intermediate_embeds = x_embeds.data.clone().detach()
            intermediate_embeds.add_(
                (1 / args.admm_rho) * lambda_embeds.data.clone().detach()
            )
            if args.conversion_method == "topk":
                _, z_ids = get_topk_closest_tokens(
                    intermediate_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    model,
                    tokenizer,
                    get_prefix(
                        args, few_shot_seqs, few_shot_labels, label_text=text_labels[0]
                    ),
                    prompt_len,
                    include_prefix=args.include_prefix,
                    topk=args.topk,
                )
            elif args.conversion_method == "concat":
                _, z_ids = sample_sequence(
                    intermediate_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    model,
                    tokenizer,
                    get_prefix(
                        args, few_shot_seqs, few_shot_labels, label_text=text_labels[0]
                    ),
                    prompt_len,
                    include_prefix=args.include_prefix,
                )
            else:
                _, z_ids = get_closest_tokens(
                    intermediate_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    metric="l2",
                )
            if isinstance(prompt_ids, list):
                z_ids[
                    :,
                    args.first_prompt_end_index
                    - prompt_len[0] : args.first_prompt_end_index,
                ] = prompt_ids[0]
                z_ids[:, -prompt_len[1] :] = prompt_ids[1]
            else:
                # print("z_ids.shape", z_ids.shape)
                # print("prompt_ids.shape", prompt_ids.shape)
                z_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
            z_embeds.data[:] = lm_embeddings(z_ids.unsqueeze(0)).detach().clone()

            def closure():
                opt.zero_grad()
                rec_loss = get_reconstruction_loss(
                    model,
                    x_embeds,
                    attention_mask,
                    true_labels_tokenized,
                    true_grads,
                    args,
                    create_graph=True,
                    previous_grad=previous_grad,
                )
                norm_diff = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
                
                if args.reg_loss_type == "norm":
                    embed_diff = norm_diff
                elif args.reg_loss_type == "embed":
                    if args.embed_loss == "cos":
                        embed_diff = 1 - cos_sim(x_embeds.mean(dim=1), avg_embeds)
                    elif args.embed_loss == "dlg":
                        embed_diff = (x_embeds.mean(dim=1) - avg_embeds).square().sum()
                    else:
                        # No regularization
                        embed_diff = torch.zeros(1)
                else:
                    embed_diff = torch.zeros(1)
                
                reg_loss = (
                    (x_embeds - z_embeds + (1 / args.admm_rho) * lambda_embeds)
                    .square()
                    .sum()
                )
                # perplexity loss
                perp_loss = get_perplexity_loss(x_embeds, z_ids, model)
                # one step update of x
                tot_loss = (
                    rec_loss
                    + (args.admm_rho / 2) * reg_loss
                    + args.coeff_reg * embed_diff
                    + args.coeff_perplexity * perp_loss
                )
                tot_loss.backward()
                # print(f"----Inner ADMM step: total loss {tot_loss.item()}, rec_loss {rec_loss.item()}, admm_item {(args.admm_rho / 2) * reg_loss.item()}, embed_diff {args.coeff_reg * embed_diff}, perp loss {args.coeff_perplexity * perp_loss}")
                if args.dataset in ["rotten_tomatoes", "imdb", "rtpolarity"]:
                    x_embeds.grad[:, -prompt_len:, :] = 0.0
                
                with torch.no_grad():
                    if args.grad_clip is not None:
                        grad_norm = x_embeds.grad.norm()  # pytype: disable=attribute-error
                        if grad_norm > args.grad_clip:
                            x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))  # pytype: disable=attribute-error

                return tot_loss, rec_loss, reg_loss, norm_diff, embed_diff, perp_loss

            for _ in range(args.admm_inner_steps):
                error, rec_loss, reg_loss, norm_diff, embed_diff, perp_loss = opt.step(
                    closure
                )
            
            # update lambda embeddings
            if args.conversion_method == "topk":
                _, proj_ids = get_topk_closest_tokens(
                    x_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    model,
                    tokenizer,
                    get_prefix(
                        args, few_shot_seqs, few_shot_labels, label_text=text_labels[0]
                    ),
                    prompt_len,
                    include_prefix=args.include_prefix,
                    topk=args.topk,
                )
            elif args.conversion_method == "concat":
                _, proj_ids = sample_sequence(
                    x_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    model,
                    tokenizer,
                    get_prefix(
                        args, few_shot_seqs, few_shot_labels, label_text=text_labels[0]
                    ),
                    prompt_len,
                    include_prefix=args.include_prefix,
                )
            else:
                _, proj_ids = get_closest_tokens(
                    x_embeds,
                    unused_tokens,
                    lm_embeddings_weight,
                    metric="l2",
                )
            if isinstance(prompt_ids, list):
                proj_ids[
                    :,
                    args.first_prompt_end_index
                    - prompt_len[0] : args.first_prompt_end_index,
                ] = prompt_ids[0]
                proj_ids[:, -prompt_len[1] :] = prompt_ids[1]
            else:
                proj_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
            
            loss_dict = get_loss(
                args,
                model,
                proj_ids,
                x_embeds,
                torch.ones(
                    proj_ids.shape[0], proj_ids.shape[1], device=device
                ).long(),  # shape: (bs, gen_tokens)
                true_labels_tokenized,
                true_grads,
                avg_embeds,
                previous_grad=previous_grad,
            )
            print(
                f"--ADMM DEBUG: iter {it} |x - z|^2 ="
                f" {(x_embeds - z_embeds).square().sum().item()}, rec_loss ="
                f" {rec_loss.item()}, reg_loss = {reg_loss.item()},"
                f" embed_loss = {embed_diff.item()}, tot_loss = {error.item()}"
                f" perp_loss = {perp_loss.item()}"
                f" rec_loss_embeds = {loss_dict['rec_loss_embeds'].item()}"
                f" rec_loss_ids = {loss_dict['rec_loss_ids'].item()}"
                f" perplexity = {loss_dict['perplexity'].item()}"
            )
            lambda_embeds.add_(
                args.admm_rho
                * (x_embeds.data.detach().clone() - z_embeds.data.detach().clone())
            )
        else:
            def closure():
                opt.zero_grad()
                rec_loss = get_reconstruction_loss(
                    model,
                    x_embeds,
                    attention_mask,
                    true_labels_tokenized,
                    true_grads,
                    args,
                    create_graph=True,
                    previous_grad=previous_grad,
                )
                norm_diff = (x_embeds.norm(p=2, dim=2).mean() - args.init_size).square()
                if args.embed_loss == "cos":
                    embed_diff = 1 - cos_sim(x_embeds.mean(dim=1), avg_embeds)
                elif args.embed_loss == "cos_mapped_embeds":
                    if args.conversion_method == "topk":
                        _, proj_ids = get_topk_closest_tokens(
                            x_embeds,
                            unused_tokens,
                            lm_embeddings_weight,
                            model,
                            tokenizer,
                            get_prefix(
                                args,
                                few_shot_seqs,
                                few_shot_labels,
                                label_text=text_labels[0],
                            ),
                            prompt_len,
                            include_prefix=args.include_prefix,
                            topk=args.topk,
                        )
                    elif args.conversion_method == "concat":
                        _, proj_ids = sample_sequence(
                            x_embeds,
                            unused_tokens,
                            lm_embeddings_weight,
                            model,
                            tokenizer,
                            get_prefix(
                                args,
                                few_shot_seqs,
                                few_shot_labels,
                                label_text=text_labels[0],
                            ),
                            prompt_len,
                            include_prefix=args.include_prefix,
                        )
                    else:
                        _, proj_ids = get_closest_tokens(
                            x_embeds,
                            unused_tokens,
                            lm_embeddings_weight,
                            metric="l2",
                        )
                    if isinstance(prompt_ids, list):
                        proj_ids[
                            :,
                            args.first_prompt_end_index
                            - prompt_len[0] : args.first_prompt_end_index,
                        ] = prompt_ids[0]
                        proj_ids[:, -prompt_len[1] :] = prompt_ids[1]
                    else:
                        proj_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
                    # print("Calulating mapped true embeds!")
                    mapped_true_embeds = lm_embeddings(proj_ids)
                    # perplexity loss
                    perp_loss = get_perplexity_loss(x_embeds, proj_ids, model)
                    cos_sim_reg = 1 - (x_embeds * mapped_true_embeds).sum() / (
                        x_embeds.norm(p=2) * mapped_true_embeds.norm(p=2)
                    )
                    embed_diff = cos_sim_reg + args.coeff_perplexity * perp_loss
                    # print(
                    #     f"embed_diff: {embed_diff.item()}, perp_loss: {perp_loss.item()},"
                    #     f" cos_sim_reg: {cos_sim_reg.item()}"
                    # )
                elif args.embed_loss == "dlg":
                    embed_diff = (x_embeds.mean(dim=1) - avg_embeds).square().sum()
                else:
                    # Default case
                    embed_diff = torch.zeros_like(norm_diff)
                
                if args.reg_loss_type == "norm":
                    reg_loss = norm_diff
                elif args.reg_loss_type == "embed":
                    reg_loss = embed_diff
                else:
                    reg_loss = 0.0
                # print(
                #     f"rec_loss: {rec_loss.item()} norm_diff:"
                #     f" {norm_diff.item()} embed_diff: {embed_diff.item()}"
                # )
                tot_loss = rec_loss + args.coeff_reg * reg_loss
                tot_loss.backward(retain_graph=True)
                with torch.no_grad():
                    if args.grad_clip is not None:
                        grad_norm = x_embeds.grad.norm()  # pytype: disable=attribute-error
                        if grad_norm > args.grad_clip:
                            x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))  # pytype: disable=attribute-error
                return tot_loss, norm_diff, embed_diff, rec_loss, reg_loss

            error, norm_diff, embed_diff, rec_loss, reg_loss = opt.step(closure)

        if isinstance(prompt_embeddings, list):
            # First prompt
            x_embeds.data[
                :,
                args.first_prompt_end_index
                - prompt_len[0] : args.first_prompt_end_index,
                :,
            ] = (
                prompt_embeddings[0].detach().clone()
            )
            # Second prompt
            x_embeds.data[:, -prompt_len[1] :, :] = (
                prompt_embeddings[1].detach().clone()
            )
        else:
            x_embeds.data[:, -prompt_len:, :] = prompt_embeddings.detach().clone()
        if best_final_error is None or error <= best_final_error:
            best_final_error = error.item()
            best_norm_diff = norm_diff.item() if norm_diff is not None else 0.0
            best_embed_diff = embed_diff.item() if embed_diff is not None else 0.0
            best_rec_loss = rec_loss.item()
            best_reg_loss = reg_loss.item()
            best_final_x.data[:] = x_embeds.data[:]
        del error, norm_diff, embed_diff, rec_loss, reg_loss

        lr_scheduler.step()

        steps_done = it + 1
        if steps_done % args.print_every == 0:
            step_time = time.time() - t_start

            if args.print_full:
                if args.conversion_method == "topk":
                    # For printing use the default prefix only
                    _, cos_ids = get_topk_closest_tokens(
                        x_embeds,
                        unused_tokens,
                        lm_embeddings_weight,
                        model,
                        tokenizer,
                        get_prefix(
                            args,
                            few_shot_seqs,
                            few_shot_labels,
                            label_text=text_labels[0],
                        ),
                        prompt_len,
                        include_prefix=args.include_prefix,
                        topk=args.topk,
                    )
                    if isinstance(prompt_ids, list):
                        cos_ids[
                            :,
                            args.first_prompt_end_index
                            - prompt_len[0] : args.first_prompt_end_index,
                        ] = prompt_ids[0]
                        cos_ids[:, -prompt_len[1] :] = prompt_ids[1]
                    else:
                        cos_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
                elif args.conversion_method == "concat":
                    # For printing use the default prefix only
                    _, cos_ids = sample_sequence(
                        x_embeds,
                        unused_tokens,
                        lm_embeddings_weight,
                        model,
                        tokenizer,
                        get_prefix(
                            args,
                            few_shot_seqs,
                            few_shot_labels,
                            label_text=text_labels[0],
                        ),
                        prompt_len,
                        include_prefix=args.include_prefix,
                    )
                    if isinstance(prompt_ids, list):
                        cos_ids[
                            :,
                            args.first_prompt_end_index
                            - prompt_len[0] : args.first_prompt_end_index,
                        ] = prompt_ids[0]
                        cos_ids[:, -prompt_len[1] :] = prompt_ids[1]
                    else:
                        cos_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
                else:
                    _, cos_ids = get_closest_tokens(
                        x_embeds, unused_tokens, lm_embeddings_weight
                    )
                    loss_dict = get_loss(
                        args,
                        model,
                        cos_ids,
                        x_embeds,
                        torch.ones(
                            cos_ids.shape[0], cos_ids.shape[1], device=device
                        ).long(),  # shape: (bs, gen_tokens)
                        true_labels_tokenized,
                        true_grads,
                        avg_embeds,
                        previous_grad=previous_grad,
                    )
                perplexity = loss_dict["perplexity"]
                rec_loss_embeds = loss_dict["rec_loss_embeds"]
                rec_loss_ids = loss_dict["rec_loss_ids"]
                tot_loss = loss_dict["tot_loss"]
                embed_diff_ids = loss_dict["embed_diff_ids"]

                print(
                    "[%4d/%4d] best_final_error=%.3f, best_rec_loss=%.3f,"
                    " best_reg_loss=%.3f,  norm_diff=%.3f,"
                    " embed_diff=%.3f, tot_loss=%.3f (perp=%.3f, rec_embeds=%.3f,"
                    " rec_ids=%.3f) embed_diff_ids=%.3f [t=%.2fs]"
                    % (
                        steps_done,
                        args.n_steps,
                        best_final_error,
                        best_rec_loss,
                        best_reg_loss,
                        best_norm_diff,
                        best_embed_diff,
                        tot_loss.item(),
                        perplexity.item(),
                        rec_loss_embeds.item(),
                        rec_loss_ids.item(),
                        embed_diff_ids.item(),
                        # tot_loss_proj.item(),
                        step_time,
                    ),
                    flush=True,
                )
                print(
                    "generation: %s" % (tokenizer.batch_decode(cos_ids)),
                    flush=True,
                )
            else:
                print(
                    "[%4d/%4d] best_final_error=%.3f, best_norm_diff=%.3f,"
                    " best_embed_diff=%.3f [t=%.2fs]"
                    % (
                        steps_done,
                        args.n_steps,
                        best_final_error,
                        best_norm_diff,
                        best_embed_diff,
                        step_time,
                    ),
                    flush=True,
                )

    # Postprocess
    if only_init:
        args.n_steps = prev_n_steps
    x_embeds.data = best_final_x

    avg_new_grad = None
    list_return_dict = []
    for default_prefix in list_prefix:
        return_dict = {}
        if args.conversion_method == "topk":
            _, cos_ids = get_topk_closest_tokens(
                x_embeds,
                unused_tokens,
                lm_embeddings_weight,
                model,
                tokenizer,
                get_prefix(
                    args,
                    few_shot_seqs,
                    few_shot_labels,
                    default_prefix,
                    label_text=text_labels[0],
                ),
                prompt_len,
                include_prefix=args.include_prefix,
                topk=args.topk,
            )
            if isinstance(prompt_ids, list):
                cos_ids[
                    :,
                    args.first_prompt_end_index
                    - prompt_len[0] : args.first_prompt_end_index,
                ] = prompt_ids[0]
                cos_ids[:, -prompt_len[1] :] = prompt_ids[1]
            else:
                cos_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
            attention_mask = torch.ones(
                gen_shape[0], cos_ids.shape[0], device=device
            ).long()  # shape: (bs, gen_tokens)
        elif args.conversion_method == "concat":
            _, cos_ids = sample_sequence(
                x_embeds,
                unused_tokens,
                lm_embeddings_weight,
                model,
                tokenizer,
                get_prefix(
                    args,
                    few_shot_seqs,
                    few_shot_labels,
                    default_prefix,
                    label_text=text_labels[0],
                ),
                prompt_len,
                include_prefix=args.include_prefix,
            )
            if isinstance(prompt_ids, list):
                cos_ids[
                    :,
                    args.first_prompt_end_index
                    - prompt_len[0] : args.first_prompt_end_index,
                ] = prompt_ids[0]
                cos_ids[:, -prompt_len[1] :] = prompt_ids[1]
            else:
                cos_ids[:, -prompt_len:] = prompt_ids  # shape: (gen_tokens,)
            attention_mask = torch.ones(
                gen_shape[0], cos_ids.shape[0], device=device
            ).long()  # shape: (bs, gen_tokens)
        else:
            _, cos_ids = get_closest_tokens(
                x_embeds, unused_tokens, lm_embeddings_weight
            )
            # cos_ids = cos_ids * attention_mask  # shape: (bs, gen_tokens)

        best_ids = cos_ids
        synthetic_inputs = []

        for i in range(best_ids.shape[0]):
            synthetic_inputs += [
                remove_padding(
                    tokenizer,
                    best_ids[i],
                    first_prompt_end_index=args.first_prompt_end_index,
                )
            ]
        loss_dict = get_loss(
            args,
            model,
            best_ids,
            x_embeds,
            torch.ones(
                best_ids.shape[0], best_ids.shape[1], device=device
            ).long(),  # shape: (bs, gen_tokens)
            true_labels_tokenized,
            true_grads,
            avg_embeds,
            previous_grad=previous_grad,
            return_grads=not args.independent_gen,
        )
        return_dict["inputs"] = synthetic_inputs
        return_dict["labels"] = true_labels
        return_dict["perplexity"] = loss_dict["perplexity"].item()
        return_dict["rec_loss_embeds"] = loss_dict["rec_loss_embeds"].item()
        return_dict["rec_loss_ids"] = loss_dict["rec_loss_ids"].item()
        return_dict["tot_loss"] = loss_dict["tot_loss"].item()
        return_dict["embed_diff_ids"] = loss_dict["embed_diff_ids"].item()
        list_return_dict.append(return_dict)
        new_grad = loss_dict.get("new_grad", None)
        if new_grad is not None:
            if avg_new_grad is None:
                avg_new_grad = []
                for grad in new_grad:
                    avg_new_grad.append(grad / len(list_prefix))
            else:
                for j, grad in enumerate(new_grad):
                    avg_new_grad[j] += grad / len(list_prefix)

    return {"list_gen_dict": list_return_dict, "new_grad": avg_new_grad}


MODEL_MAP = {
    "phi": "microsoft/phi-1_5",
}

LAST_LAYERS = [
    "lm_head"
]


def get_gen_samples(args):
    """Get reference data for generation.

    Args:
        args: Arguments

    Returns:
        pos_sequences: Positive samples
        neg_sequences: Negative samples
        pos_labels: Positive labels
        neg_labels: Negative labels
    """
    dataset = TextDataset(
        args.device,
        args.dataset,
        args.split,
        args.n_gen_samples,
        1,
        n_fewshot=args.n_fewshot,
        seed=args.rng_seed,
    )
    
    pos_sequences, neg_sequences = [], []
    pos_labels, neg_labels = [], []
    fewshot_seqs, fewshot_labels = [], []
    if args.subset_size > args.n_gen_samples:
        args.subset_size = args.n_gen_samples
    for i in range(args.subset_size + args.n_fewshot):
        seq, true_label = dataset[i]
        if i < args.n_gen_samples:
            if true_label == 1:
                pos_sequences.extend(seq)
                pos_labels.extend(true_label)
            else:
                neg_sequences.extend(seq)
                neg_labels.extend(true_label)
        else:
            fewshot_seqs.extend(seq)
            fewshot_labels.extend(true_label)

    return {
        "pos_sequences": pos_sequences,
        "neg_sequences": neg_sequences,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels,
        "fewshot_seqs": fewshot_seqs,
        "fewshot_labels": fewshot_labels,
    }


def dp_cliped_per_sample_grads(args, per_sample_grads):
    """Clip per sample grads to make it norm of C."""
    # clipped grads to make it norm of C
    flat_per_sample_grads_list = []
    for grads in per_sample_grads:
        flat_per_sample_grads_list.append(grads.view(-1))
    all_flat_per_sample_grads = torch.cat(flat_per_sample_grads_list)
    per_sample_total_norms = torch.norm(all_flat_per_sample_grads, p=2)
    
    del all_flat_per_sample_grads
    clipping_factors = (args.dp_c / (per_sample_total_norms + 1e-6)).clamp(
        max=1.0
    )
    per_sample_grads = [
        grads.mul_(clipping_factors) for grads in per_sample_grads
    ]

    return per_sample_grads


def dp_add_noise(args, average_true_grads, batch_size):
    """add dp gaussian noise to average true grads."""
    if args.dp_epsilon <= 1 and args.dp_epsilon >= 0:
        for grads in average_true_grads:
            std = (
                (args.dp_c / batch_size)
                * math.sqrt(2 * math.log(1.25 / args.dp_delta))
            ) / args.dp_epsilon
            noise = torch.normal(
                mean=0,
                std=std,
                size=grads.shape,
                device=grads.device,
            )
            grads.add_(noise)
    elif args.dp_epsilon > 1:
        for grads in average_true_grads:
            c = math.sqrt(math.log((2) / ((math.sqrt(16 * args.dp_delta + 1) - 1))))
            std = (
                (c + math.sqrt(c**2 + args.dp_epsilon)) * (args.dp_c / batch_size)
            ) / (args.dp_epsilon * math.sqrt(2))
            noise = torch.normal(
                mean=0,
                std=std,
                size=grads.shape,
                device=grads.device,
            )
            grads.add_(noise)

    return average_true_grads


def compute_list_embeds(args, model, tokenizer, sequences, labels):
    """Compute average gradients.

    Args:
        args: Arguments
        model: Model
        tokenizer: Tokenizer
        sequences: List of samples
        labels: Labels

    Returns:
        list_true_embeds: List of true embeddings
        prompt_lengths: List of prompt lengths
    """
    lm_embeddings = model.get_input_embeddings()
    num_samples = len(sequences)
    text_labels = []
    prompt_lengths = []  # Collect prompt lengths for dataset with two prompts
    
    if args.dataset in [
        "sst2",
        "rotten_tomatoes",
        "imdb",
        "rtpolarity",
    ]:
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

    list_true_embeds = []

    for i in range(num_samples):
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
        list_true_embeds.append(true_embeds.detach().cpu())

    return list_true_embeds, prompt_lengths


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
        list_true_embeds: List of true embeddings
        avg_true_embeds: Average true embeddings
        closest_index: Index of the gradient closest to the average gradient
        prompt_lengths: List of prompt lengths
    """
    lm_embeddings = model.get_input_embeddings()
    num_samples = len(sequences)
    text_labels = []
    prompt_lengths = []  # Collect prompt lengths for dataset with two prompts
    
    if args.dataset in [
        "sst2",
        "rotten_tomatoes",
        "imdb",
        "rtpolarity",
    ]:
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
    list_true_embeds = []
    avg_true_embeds = None

    for i in range(num_samples):
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
            model,
            true_embeds,
            orig_batch["attention_mask"],
            label,
            gen_grad_clip=args.gen_grad_clip,
        )
        if average_grads is None:
            average_grads = []
            if args.use_dp:
                curr_grads = dp_cliped_per_sample_grads(args, curr_grads)
            for grad in curr_grads:
                average_grads.append(grad.detach() / num_samples)
        else:
            if args.use_dp:
                curr_grads = dp_cliped_per_sample_grads(args, curr_grads)
            for j, grad in enumerate(curr_grads):
                average_grads[j].add_(grad.detach() / num_samples)
        if avg_true_embeds is None:
            avg_true_embeds = true_embeds.detach().mean(dim=1) / num_samples
        else:
            avg_true_embeds.add_(true_embeds.detach().mean(dim=1) / num_samples)
        
        list_true_embeds.append(true_embeds.detach().cpu())
        del curr_grads
        torch.cuda.empty_cache()

    # Compute the closest gradient to the average gradient
    # Need to recalculate the gradients to avoid OOM
    smallest_grad_dist = 1e9
    closest_index = -1
    for i in range(num_samples):
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
            model,
            true_embeds,
            orig_batch["attention_mask"],
            label,
            gen_grad_clip=args.gen_grad_clip,
        )
        curr_grad_dist = grad_dist(average_grads, curr_grads, args)
        if curr_grad_dist < smallest_grad_dist:
            smallest_grad_dist = curr_grad_dist
            closest_index = i
        del curr_grads
        torch.cuda.empty_cache()

    # add dp gaussian noise to the gradients
    if args.use_dp:
        average_grads = dp_add_noise(args, average_grads, num_samples)

    return (
        average_grads,
        list_true_embeds,
        avg_true_embeds,
        prompt_lengths,
        closest_index,
    )


def main():
    summary_metrics = {}
    args = get_args()
    args.work_dir = os.path.join(args.work_base_dir, get_args_flags(args))
    if os.path.exists(args.work_dir):
        print("Work directory already exists: ", args.work_dir)
        if args.overwrite:
            print("Overwriting work directory...")
            shutil.rmtree(args.work_dir)
        else:
            print("Restoring RNG state ... ")
            if not os.path.exists(os.path.join(args.work_dir, "rng_states.pth")):
                print("RNG state file not found. Starting from scratch.")
                set_all_seeds(args.rng_seed)
                # key metrics
                summary_metrics["mean_perplexity"] = None
                summary_metrics["mean_rec_loss"] = None
                summary_metrics["mean_tot_loss"] = None
                summary_metrics["perplexity"] = []
                summary_metrics["rec_loss_embeds"] = []
                summary_metrics["rec_loss_ids"] = []
                summary_metrics["tot_loss"] = []
                summary_metrics["embed_diff_ids"] = []
                pos_generations = []
                neg_generations = []
            else:
                load_rng_states(args.work_dir)
                # determine remaining n_gen
                num_samples = count_lines(
                    os.path.join(args.work_dir, "synthetic_data.jsonl")
                )
                # args.n_gen = args.n_gen - num_samples // 2
                args.skip_first_samples += num_samples // (2 * args.gen_bs)
                print("Remaining n_gen: ", args.n_gen - args.skip_first_samples)
                with open(
                    os.path.join(args.work_dir, "summary_metrics.pkl"), "rb"
                ) as f:
                    summary_metrics = pickle.load(f)
                with open(
                    os.path.join(args.work_dir, "pos_generations.pkl"), "rb"
                ) as f:
                    pos_generations = pickle.load(f)
                with open(
                    os.path.join(args.work_dir, "neg_generations.pkl"), "rb"
                ) as f:
                    neg_generations = pickle.load(f)
    else:
        print("Creating work directory: ", args.work_dir)
        set_all_seeds(args.rng_seed)
        # key metrics
        summary_metrics["mean_perplexity"] = None
        summary_metrics["mean_rec_loss"] = None
        summary_metrics["mean_tot_loss"] = None
        summary_metrics["perplexity"] = []
        summary_metrics["rec_loss_embeds"] = []
        summary_metrics["rec_loss_ids"] = []
        summary_metrics["tot_loss"] = []
        summary_metrics["embed_diff_ids"] = []
        pos_generations = []
        neg_generations = []
    
    os.makedirs(args.work_dir, exist_ok=True)
    summary_metrics["args"] = vars(args)
    print("\n\n\nCommand:", " ".join(sys.argv), "\n\n\n", flush=True)
    print("Full args:", args, "\n\n\n", flush=True)
    wandb.init(
        project="gradmm",
        config=args,
        name=get_args_flags(args),
    )

    device = torch.device(args.device)
    model, tokenizer = None, None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model_name],
        device_map="auto",
    )
    # Set gradient for only last layers
    if args.last_layer_gradient:
        named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if any(substring in name for substring in LAST_LAYERS):
                named_parameters_to_optim.append((name, param))
            else:
                param.requires_grad = False

        assert len(named_parameters_to_optim) != 0, "no layer found"
        print(f"Set gradients for {len(named_parameters_to_optim)} layers")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAP[args.model_name], use_fast=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("\n\ngenerating..\n", flush=True)

    # prepare inputs
    samples_dict = get_gen_samples(args)
    pos_sequences = samples_dict["pos_sequences"]
    neg_sequences = samples_dict["neg_sequences"]
    pos_labels = samples_dict["pos_labels"]
    neg_labels = samples_dict["neg_labels"]
    fewshot_seqs = samples_dict["fewshot_seqs"]
    fewshot_labels = samples_dict["fewshot_labels"]
    summary_metrics["pos_num_samples"] = len(pos_sequences)
    summary_metrics["neg_num_samples"] = len(neg_sequences)

    with open(
        os.path.join(args.work_dir, "real_train_data.jsonl"), "w"
    ) as f:
        count = 0
        for seq, label in zip(pos_sequences, pos_labels):
            data = {"id": count, "inputs": seq, "label": label.item()}
            f.write(json.dumps(data) + "\n")
            count += 1
        for seq, label in zip(neg_sequences, neg_labels):
            data = {"id": count, "inputs": seq, "label": label.item()}
            f.write(json.dumps(data) + "\n")
            count += 1
        for seq, label in zip(fewshot_seqs, fewshot_labels):
            data = {"id": count, "inputs": seq, "label": label.item()}
            f.write(json.dumps(data) + "\n")
            count += 1

    # count token that only used in the dataset
    token_candidates = None
    if args.use_sample_tokens_only:
        token_candidates = set()
        for seq in pos_sequences:
            input_ids = tokenizer(
                seq, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].view(-1)
        for t in input_ids:
            token_candidates.add(t.item())
        for seq in neg_sequences:
            input_ids = tokenizer(
                seq, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].view(-1)
        for t in input_ids:
            token_candidates.add(t.item())
        
        prompt_seq = [
            "\nIs the sentence grammatically acceptable? Answer:\n",
            "Yes",
            "No",
            " It was ",
            "great",
            "bad",
            'Does this mean that "',
            '" is true? Yes or No?',
            "",
        ]
        for seq in prompt_seq:
            input_ids = tokenizer(
                seq, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"].view(-1)
        for t in input_ids:
            token_candidates.add(t.item())

    # extract unique words from the dataset
    if args.prefix_option == "random":
        assert (
            args.conversion_method == "topk"
        ), "Prefix option random requires topk decoding."
        unique_prefixes = set()
        unique_prefixes.update(extract_first_words(pos_sequences))
        unique_prefixes.update(extract_first_words(neg_sequences))
        unique_prefixes.update(extract_first_words(fewshot_seqs))
        print(f"Unique words in the dataset: {len(unique_prefixes)}")
        unique_prefixes = random.sample(list(unique_prefixes), args.n_prefix)
    else:
        unique_prefixes = ["The"]

    # Create data loaders
    pos_data_loader = BatchDatasetLoader(
        pos_sequences, pos_labels, args.batch_size
    )
    neg_data_loader = BatchDatasetLoader(
        neg_sequences, neg_labels, args.batch_size
    )

    pos_previous_grad = None
    neg_previous_grad = None
    pos_init_sequences = []
    neg_init_sequences = []
    (
        pos_true_grads,
        pos_true_embeds,
        pos_avg_embeds,
        pos_prompt_lengths,
        pos_closest_index,
    ) = (None, None, None, None, None)
    (
        neg_true_grads,
        neg_true_embeds,
        neg_avg_embeds,
        neg_prompt_lengths,
        neg_closest_index,
    ) = (None, None, None, None, None)

    for i in range(args.n_gen):
        print(f"Generate input #{i} of {args.n_gen}.")
        # Sample a mini-batch of real data
        pos_sequences, pos_labels = next(pos_data_loader)
        neg_sequences, neg_labels = next(neg_data_loader)
        # Calculate average gradients & embeddings
        if pos_true_grads is None or args.batch_size < summary_metrics["pos_num_samples"]:
            print("Calculating average gradients for positive samples.")
            (
                pos_true_grads,
                pos_true_embeds,
                pos_avg_embeds,
                pos_prompt_lengths,
                pos_closest_index,
            ) = compute_average_grads(
                args, model, tokenizer, pos_sequences, pos_labels
            )
        else:
            pos_true_embeds, pos_prompt_lengths = compute_list_embeds(
                args, model, tokenizer, pos_sequences, pos_labels
            )
        if neg_true_grads is None or args.batch_size < summary_metrics["neg_num_samples"]:
            print("Calculating average gradients for negative samples.")
            (
                neg_true_grads,
                neg_true_embeds,
                neg_avg_embeds,
                neg_prompt_lengths,
                neg_closest_index,
            ) = compute_average_grads(
                args, model, tokenizer, neg_sequences, neg_labels
            )
        else:
            neg_true_embeds, neg_prompt_lengths = compute_list_embeds(
                args, model, tokenizer, neg_sequences, neg_labels
            )
        # Save the average gradients for the positive and negative samples.
        if args.save_avg_grad:
            with open(
                os.path.join(args.work_dir, "pos_avg_grads.pkl"), "wb"
            ) as f:
                torch.save(pos_true_grads, f)
            with open(
                os.path.join(args.work_dir, "neg_avg_grads.pkl"), "wb"
            ) as f:
                torch.save(neg_true_grads, f)
            sys.exit(0)
        
        if args.init == "real_first":
            pos_true_embed_index = 0
            neg_true_embed_index = 0
        elif args.init == "real_closest":
            pos_true_embed_index = pos_closest_index
            neg_true_embed_index = neg_closest_index
        else:
            pos_true_embed_index = np.random.randint(len(pos_true_embeds))
            neg_true_embed_index = np.random.randint(len(neg_true_embeds))
        
        # Initialization
        pos_init = pos_sequences[pos_true_embed_index]
        pos_init_embed = pos_true_embeds[pos_true_embed_index]
        pos_init_length = pos_prompt_lengths[pos_true_embed_index]
        neg_init = neg_sequences[neg_true_embed_index]
        neg_init_embed = neg_true_embeds[neg_true_embed_index]
        neg_init_length = neg_prompt_lengths[neg_true_embed_index]
        print(
            f"Real pos init: {pos_init} length = {pos_init_length} and embed shape"
            f" of {pos_init_embed.shape}"
        )
        print(
            f"Real neg init: {neg_init} length = {neg_init_length} and embed shape"
            f" of {neg_init_embed.shape}"
        )
        pos_init_sequences.append(pos_sequences[pos_true_embed_index])
        neg_init_sequences.append(neg_sequences[neg_true_embed_index])
        print("Positive average sequence length", np.mean(pos_prompt_lengths))
        
        # Positive generation
        if args.use_auto_gen_tokens:
            args.gen_max_tokens = int(np.mean(pos_prompt_lengths))
            print("Setting gen_max_tokens to", args.gen_max_tokens)
        pos_gen = generation(
            args,
            device,
            None,
            pos_true_grads,
            pos_labels[:1],
            pos_init_embed,
            pos_init_length,
            pos_avg_embeds,
            fewshot_seqs,
            fewshot_labels,
            tokenizer,
            model,
            token_candidates,
            unique_prefixes,
            only_init=True if i < args.skip_first_samples else False,
            previous_grad=pos_previous_grad,
        )
        # Negative generation
        print("Negative average sequence length", np.mean(neg_prompt_lengths))
        if args.use_auto_gen_tokens:
            args.gen_max_tokens = int(np.mean(neg_prompt_lengths))
            print("Setting gen_max_tokens to", args.gen_max_tokens)
        neg_gen = generation(
            args,
            device,
            None,
            neg_true_grads,
            neg_labels[:1],
            neg_init_embed,
            neg_init_length,
            neg_avg_embeds,
            fewshot_seqs,
            fewshot_labels,
            tokenizer,
            model,
            token_candidates,
            unique_prefixes,
            only_init=True if i < args.skip_first_samples else False,
            previous_grad=neg_previous_grad,
        )
        if i < args.skip_first_samples:
            continue
        # Update previous gradients if not independent generations
        if not args.independent_gen:
            print("Update previous grad")
            if pos_previous_grad is None:
                pos_previous_grad = pos_gen["new_grad"]
                neg_previous_grad = neg_gen["new_grad"]
            else:
                for j, grad in enumerate(pos_gen["new_grad"]):
                    pos_previous_grad[j] += grad[j]
                for j, grad in enumerate(neg_gen["new_grad"]):
                    neg_previous_grad[j] += grad[j]

        # Collect generations
        list_pos_gen = pos_gen["list_gen_dict"]
        list_neg_gen = neg_gen["list_gen_dict"]
        pos_generations.extend(list_pos_gen)
        neg_generations.extend(list_neg_gen)
        print(f"Done with input #{i} of {args.n_gen}.")
        for pos_gen, neg_gen in zip(list_pos_gen, list_neg_gen):
            for seq in pos_gen["inputs"]:
                print("=========Pos============")
                print(seq)
                print("========================")

            print("predicted: ")
            for seq in neg_gen["inputs"]:
                print("=========Neg========")
                print(seq)
                print("========================")
            # print(pos_gen, neg_gen)
        summary_metrics["perplexity"].append(pos_gen["perplexity"])
        summary_metrics["perplexity"].append(neg_gen["perplexity"])
        summary_metrics["rec_loss_embeds"].append(pos_gen["rec_loss_embeds"])
        summary_metrics["rec_loss_embeds"].append(neg_gen["rec_loss_embeds"])
        summary_metrics["rec_loss_ids"].append(pos_gen["rec_loss_ids"])
        summary_metrics["rec_loss_ids"].append(neg_gen["rec_loss_ids"])
        summary_metrics["tot_loss"].append(pos_gen["tot_loss"])
        summary_metrics["tot_loss"].append(neg_gen["tot_loss"])
        summary_metrics["embed_diff_ids"].append(pos_gen["embed_diff_ids"])
        summary_metrics["embed_diff_ids"].append(neg_gen["embed_diff_ids"])
        if args.save_every > 0 and i % args.save_every == 0:
            save_rng_states(args.work_dir)
            # save pos_generations and neg_generations and summary_metrics to pkl
            with open(
                os.path.join(args.work_dir, "pos_generations.pkl"), "wb"
            ) as f:
                pickle.dump(pos_generations, f)
            with open(
                os.path.join(args.work_dir, "neg_generations.pkl"), "wb"
            ) as f:
                pickle.dump(neg_generations, f)
            with open(
                os.path.join(args.work_dir, "summary_metrics.pkl"), "wb"
            ) as f:
                pickle.dump(summary_metrics, f)
            with open(
                os.path.join(args.work_dir, "synthetic_data.jsonl"), "w"
            ) as f:
                count = 0
                for gen in pos_generations:
                    if type(gen["inputs"]) == list:
                        for seq in gen["inputs"]:
                            data = {
                                "id": count,
                                "inputs": seq,
                                "label": pos_labels[0].item(),
                                "tot_loss": gen["tot_loss"],
                                "perplexity": gen["perplexity"],
                                "rec_loss_embeds": gen["rec_loss_embeds"],
                                "rec_loss_ids": gen["rec_loss_ids"],
                                "embed_diff_ids": gen["embed_diff_ids"],
                            }
                            f.write(json.dumps(data) + "\n")
                            count += 1
                    else:
                        data = {
                            "id": count,
                            "inputs": gen["inputs"],
                            "label": pos_labels[0].item(),
                            "tot_loss": gen["tot_loss"],
                            "perplexity": gen["perplexity"],
                            "rec_loss_embeds": gen["rec_loss_embeds"],
                            "rec_loss_ids": gen["rec_loss_ids"],
                            "embed_diff_ids": gen["embed_diff_ids"],
                        }
                        f.write(json.dumps(data) + "\n")
                        count += 1
                for gen in neg_generations:
                    if type(gen["inputs"]) == list:
                        for seq in gen["inputs"]:
                            data = {
                                "id": count,
                                "inputs": seq,
                                "label": neg_labels[0].item(),
                                "tot_loss": gen["tot_loss"],
                                "perplexity": gen["perplexity"],
                                "rec_loss_embeds": gen["rec_loss_embeds"],
                                "rec_loss_ids": gen["rec_loss_ids"],
                                "embed_diff_ids": gen["embed_diff_ids"],
                            }
                            f.write(json.dumps(data) + "\n")
                            count += 1
                    else:
                        data = {
                            "id": count,
                            "inputs": gen["inputs"],
                            "label": neg_labels[0].item(),
                            "tot_loss": gen["tot_loss"],
                            "perplexity": gen["perplexity"],
                            "rec_loss_embeds": gen["rec_loss_embeds"],
                            "rec_loss_ids": gen["rec_loss_ids"],
                            "embed_diff_ids": gen["embed_diff_ids"],
                        }
                        f.write(json.dumps(data) + "\n")
                        count += 1
    # Save summary metrics
    summary_metrics["mean_perplexity"] = np.mean(summary_metrics["perplexity"])
    summary_metrics["mean_rec_loss_embeds"] = np.mean(
        summary_metrics["rec_loss_embeds"]
    )
    summary_metrics["mean_rec_loss_ids"] = np.mean(
        summary_metrics["rec_loss_ids"]
    )
    summary_metrics["mean_tot_loss"] = np.mean(summary_metrics["tot_loss"])
    summary_metrics["mean_embed_diff_ids"] = np.mean(
        summary_metrics["embed_diff_ids"]
    )
    with open(
        os.path.join(args.work_dir, "summary_metrics.json"), "w"
    ) as f:
        print("Writing summary metrics...")
        f.write(json.dumps(summary_metrics, indent=2))
    with open(
        os.path.join(args.work_dir, "synthetic_data.jsonl"), "w"
    ) as f:
        count = 0
        for gen in pos_generations:
            if type(gen["inputs"]) == list:
                for seq in gen["inputs"]:
                    data = {
                        "id": count,
                        "inputs": seq,
                        "label": pos_labels[0].item(),
                        "tot_loss": gen["tot_loss"],
                        "perplexity": gen["perplexity"],
                        "rec_loss_embeds": gen["rec_loss_embeds"],
                        "rec_loss_ids": gen["rec_loss_ids"],
                        "embed_diff_ids": gen["embed_diff_ids"],
                    }
                    f.write(json.dumps(data) + "\n")
                    count += 1
            else:
                data = {
                    "id": count,
                    "inputs": gen["inputs"],
                    "label": pos_labels[0].item(),
                    "tot_loss": gen["tot_loss"],
                    "perplexity": gen["perplexity"],
                    "rec_loss_embeds": gen["rec_loss_embeds"],
                    "rec_loss_ids": gen["rec_loss_ids"],
                    "embed_diff_ids": gen["embed_diff_ids"],
                }
                f.write(json.dumps(data) + "\n")
                count += 1
        for gen in neg_generations:
            if type(gen["inputs"]) == list:
                for seq in gen["inputs"]:
                    data = {
                        "id": count,
                        "inputs": seq,
                        "label": neg_labels[0].item(),
                        "tot_loss": gen["tot_loss"],
                        "perplexity": gen["perplexity"],
                        "rec_loss_embeds": gen["rec_loss_embeds"],
                        "rec_loss_ids": gen["rec_loss_ids"],
                        "embed_diff_ids": gen["embed_diff_ids"],
                    }
                    f.write(json.dumps(data) + "\n")
                    count += 1
            else:
                data = {
                    "id": count,
                    "inputs": gen["inputs"],
                    "label": neg_labels[0].item(),
                    "tot_loss": gen["tot_loss"],
                    "perplexity": gen["perplexity"],
                    "rec_loss_embeds": gen["rec_loss_embeds"],
                    "rec_loss_ids": gen["rec_loss_ids"],
                    "embed_diff_ids": gen["embed_diff_ids"],
                }
                f.write(json.dumps(data) + "\n")
                count += 1
    # Save real init data
    with open(
        os.path.join(args.work_dir, "real_init_data.jsonl"), "w"
    ) as f:
        count = 0
        for seq in pos_init_sequences:
            data = {"id": count, "inputs": seq, "label": 1}
            f.write(json.dumps(data) + "\n")
            count += 1
        for seq in neg_init_sequences:
            data = {"id": count, "inputs": seq, "label": 0}
            f.write(json.dumps(data) + "\n")
            count += 1


if __name__ == "__main__":
    main()
