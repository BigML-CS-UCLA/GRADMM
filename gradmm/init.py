import torch

from utilities import (
    get_closest_tokens,
    get_reconstruction_loss,
    get_reconstruction_loss_ids
)


def get_init_lm(
    args,
    model,
    unused_tokens,
    shape,
    prompt_embeddings,
    attention_mask,
    true_labels,
    true_grads,
    lm_embeddings_weight,
    tokenizer,
    previous_grad=None,
):
    """Get initial embeddings.

    Args:
        args: Arguments
        model: Model
        unused_tokens: List of unused tokens
        shape: Shape of the initial embeddings
        prompt_embeddings: Task prompt embeddings
        attention_mask: Attention mask
        true_labels: True labels
        true_grads: True gradients
        lm_embeddings_weight: Model embeddings weight
        tokenizer: Tokenizer
        previous_grad: Previous gradients

    Returns:
        x_embeds: Initial embeddings
    """
    device = model.device
    num_inits = shape[0]
    if isinstance(prompt_embeddings, list):
        prompt_len = [prompt.shape[0] for prompt in prompt_embeddings]
        max_new_tokens = shape[1] // 2
    else:
        prompt_len = prompt_embeddings.shape[0]
        max_new_tokens = shape[1]

    # Generate candidates from random
    new_shape = [args.init_candidates * num_inits] + list(shape[1:])
    embeds = torch.randn(new_shape).to(device)
    if args.init == 'random_normal':
        embeds = torch.randn(new_shape).to(device)
    elif args.init == 'random_embed':
        # Randomly sample word indices
        random_indices = torch.randint(
            0, lm_embeddings_weight.shape[1], new_shape[:2]
        )
        embeds = lm_embeddings_weight.squeeze(0)[random_indices]

    # fix prompt embeddings
    if isinstance(prompt_embeddings, list):
        # First prompt
        embeds.data[:, max_new_tokens - prompt_len[0] : max_new_tokens, :] = (
            prompt_embeddings[0]
        )
        # Second prompt
        embeds.data[:, -prompt_len[1] :, :] = prompt_embeddings[1]
    else:
        embeds.data[:, -prompt_len:, :] = prompt_embeddings

    # Pick candidates based on rec loss
    best_x_embeds, best_rec_loss = None, None
    for i in range(args.init_candidates):
        tmp_embeds = embeds[i * num_inits : (i + 1) * num_inits]

        rec_loss = get_reconstruction_loss(
            model,
            tmp_embeds,
            attention_mask,
            true_labels,
            true_grads,
            args,
            previous_grad=previous_grad,
        )
        if (best_rec_loss is None) or (rec_loss < best_rec_loss):
            best_rec_loss = rec_loss
            best_x_embeds = tmp_embeds
            _, cos_ids = get_closest_tokens(
                tmp_embeds, unused_tokens, lm_embeddings_weight, metric='cos'
            )
            best_rec_loss_ids = get_reconstruction_loss_ids(
                model,
                cos_ids,
                attention_mask,
                true_labels,
                true_grads,
                args,
            )
            cos_ids = cos_ids * attention_mask
            sen = tokenizer.batch_decode(cos_ids)
            print(
                f'[Init] best rec_loss_embeds: {best_rec_loss.item()} best'
                f' rec_loss_ids: {best_rec_loss_ids.item()} for {sen}',
                flush=True,
            )

    # Scale inital embeddings to args.init_size
    # (e.g., avg of BERT embeddings ~1.4)
    if args.init_size >= 0:
        best_x_embeds /= torch.norm(best_x_embeds, dim=2, keepdim=True)
        best_x_embeds *= args.init_size

    x_embeds = best_x_embeds.detach().clone()
    x_embeds.requires_grad_(True)

    return x_embeds
