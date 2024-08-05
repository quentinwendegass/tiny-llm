import sys

import torch

from helper.helper import load_checkpoint_state
from helper.tokenization import load_tokenizer


def generate_text(
    model,
    tokenizer,
    start_prompt,
    max_length=64,
    temperature=1.0,
    top_k=50,
    device=torch.device("cpu"),
):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(start_prompt).ids]).to(device)
    with torch.no_grad():
        for _ in range(max_length - len(input_ids[0])):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                # Top-K sampling
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token_id = torch.multinomial(top_k_probs, 1)
                next_token = top_k_indices.gather(1, next_token_id)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.squeeze().unsqueeze(0).unsqueeze(0)
            print(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break

    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Please execute with name and path to checkpoint as arguments")
        sys.exit(1)

    device = torch.device("cpu")

    checkpoint_path = sys.argv[2]
    name = sys.argv[1]
    checkpoint = load_checkpoint_state(checkpoint_path, device)

    prompt = "\n"

    generated_text = generate_text(
        checkpoint[0],
        load_tokenizer(name),
        prompt,
        max_length=100,
        temperature=1,
        top_k=50,
        device=device,
    )

    print("Generated Text:")
    print(generated_text)
