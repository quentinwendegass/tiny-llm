import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from generate import generate_text
from helper.helper import load_checkpoint_state
from helper.tokenization import load_tokenizer

device = torch.device("cpu")
checkpoint = load_checkpoint_state("configurations/tiny-stories/checkpoint.pt", device)
context_length = checkpoint[5]["context_len"]
model = checkpoint[0]
tokenizer = load_tokenizer("tiny-stories")
max_input_len = 2000

app = FastAPI()


@app.get("/status")
async def status():
    return {"status": "OK"}


@app.get("/generate")
async def generate(
    top_k: int = 50,
    top_p: float = 1,
    temperature: float = 1,
    max_length: int = 100,
    input_text: str = "\n",
    repetition_penalty: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
):
    if input_text == "":
        input_text = "\n"

    input_text = input_text[-context_length:]

    if max_length > max_input_len:
        max_length = max_input_len

    return StreamingResponse(
        generate_text(
            model,
            tokenizer,
            input_text,
            context_length,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ),
        media_type="text/event-stream",
    )
