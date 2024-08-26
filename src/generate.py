import sys

import torch

from helper.helper import load_checkpoint_state
from helper.tokenization import load_tokenizer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = filter_value
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(torch.nn.functional.softmax(logits, -1), 1)

    return pred_token


def generate_text(
    model,
    tokenizer,
    start_prompt,
    context_len,
    max_length=64,
    temperature=1,
    top_k=50,
    top_p=1,
    repetition_penalty=1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    device=torch.device("cpu"),
):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(start_prompt).ids]).to(device)

    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in torch.unique(input_ids):
                    logits[:, token_id] /= repetition_penalty

            # Apply presence and frequency penalties
            if presence_penalty != 0.0 or frequency_penalty != 0.0:
                token_counts = torch.bincount(
                    input_ids.view(-1), minlength=logits.size(-1)
                )
                presence_mask = (token_counts > 0).float()
                frequency_mask = token_counts.float()

                logits -= presence_penalty * presence_mask
                logits -= frequency_penalty * frequency_mask

            # Apply top-k and top-p sampling
            next_token = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            yield tokenizer.decode([next_token])

            if input_ids.size(1) >= context_len:
                input_ids = input_ids[:, 1:]
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.token_to_id("</s>"):
                break


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Please execute with name and path to checkpoint as arguments")
        sys.exit(1)

    device = torch.device("cpu")

    checkpoint_path = sys.argv[2]
    name = sys.argv[1]
    checkpoint = load_checkpoint_state(checkpoint_path, device)

    prompt = "\n"

    test_generator = generate_text(
        checkpoint[0],
        load_tokenizer(name),
        prompt,
        checkpoint[5]["context_len"],
        max_length=200,
        device=device,
    )

    for token in test_generator:
        print(token, end="")
