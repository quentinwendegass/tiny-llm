import argparse
import logging
import math

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from helper.dataloader import create_data_loaders
from helper.helper import (
    create_new_state,
    load_checkpoint_state,
    load_params,
    save_checkpoint,
)
from helper.tokenization import load_tokenizer


def start_training(
    name,
    num_epochs,
    checkpoint=None,
    device_type=None,
    use_wandb=True,
):
    if device_type is None:
        device_type = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_type)
    logging.info(f"Using device: {device}")

    params = load_params(name)
    tokenizer = load_tokenizer(name)
    train_dataloader, val_dataloader = create_data_loaders(
        name, params["batch_size"], params["data_in_memory"]
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    training_steps = math.floor(len(train_dataloader.dataset) / params["batch_size"])
    validation_steps = math.floor(len(val_dataloader.dataset) / params["batch_size"])
    max_steps = training_steps * num_epochs
    logging.info(f"Training for {num_epochs} epochs and {max_steps} steps")

    if use_wandb:
        wandb.init(
            project=name,
            config={
                "params": params,
                "num_epochs": num_epochs,
                "max_steps": max_steps,
                "training_steps": training_steps,
                "validation_steps": validation_steps,
                "device_type": device_type,
            },
        )

    if checkpoint:
        model, optimizer, scheduler, start_epoch, start_step, params = (
            load_checkpoint_state(checkpoint, device)
        )
    else:
        model, optimizer, scheduler, start_epoch, start_step, params = create_new_state(
            params, device, max_steps
        )

    torch.set_float32_matmul_precision("high")

    uncompiled_model = model
    if device_type == "cuda":
        logging.info("Compiling model")
        model = torch.compile(model, mode="reduce-overhead")

    if use_wandb:
        wandb.watch(model, log_freq=10)

    best_val_loss = float("inf")

    def get_loss_for_batch(x, y):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        y = y.view(-1)
        pred = pred.view(-1, params["vocab_size"])

        return loss_fn(pred, y)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        total_train_loss = 0

        for batch_idx, (texts, labels) in tqdm(
            enumerate(train_dataloader), total=training_steps
        ):
            if start_step != 0 and start_step >= batch_idx:
                continue

            loss = get_loss_for_batch(texts, labels)

            total_train_loss += loss.item()

            loss = loss / params["gradient_accumulation_steps"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_wandb:
                wandb.log(
                    {
                        "Train/Loss": loss * params["gradient_accumulation_steps"],
                        "Warmup Rate": scheduler.rate(),
                    },
                    step=(epoch - 1) * training_steps + batch_idx,
                )

            if ((batch_idx + 1) % params["gradient_accumulation_steps"] == 0) or (
                batch_idx + 1 == training_steps
            ):
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if batch_idx > 0 and batch_idx % params["validation_freq"] == 0:
                if val_dataloader:
                    model.eval()
                    total_val_loss = 0
                    with torch.no_grad():
                        for texts_val, labels_val in tqdm(
                            val_dataloader, total=validation_steps
                        ):
                            loss_val = get_loss_for_batch(texts_val, labels_val)
                            total_val_loss += loss_val.item()
                            if use_wandb:
                                wandb.log(
                                    {"Val/Loss": loss_val.item()},
                                    step=(epoch - 1) * validation_steps + batch_idx,
                                )

                    avg_val_loss = total_val_loss / validation_steps
                    avg_train_loss = total_train_loss / (batch_idx + 1)

                    if avg_val_loss < best_val_loss:
                        save_checkpoint(
                            name,
                            epoch,
                            batch_idx,
                            uncompiled_model,
                            optimizer,
                            scheduler,
                            avg_train_loss,
                            avg_val_loss,
                        )
                        best_val_loss = avg_val_loss

                    logging.info(f"Train Loss: {avg_train_loss}")
                    logging.info(f"Validation Loss: {avg_val_loss}")

                    model.train()

        logging.info(f"Epoch finished {epoch}/{num_epochs}")


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Is MPS built? {torch.backends.mps.is_built()}")
    logging.info(f"Is MPS available? {torch.backends.mps.is_available()}")
    logging.info(f"Is Cuda built? {torch.backends.cuda.is_built()}")
    logging.info(f"Is Cuda available? {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser(prog="Model training")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        required=False,
        help="Checkpoint path",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, required=False, help="Number of epochs"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=False,
        help="Device",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    start_training(
        args.model_name, args.epochs, args.checkpoint, args.device, args.wandb
    )


if __name__ == "__main__":
    main()
