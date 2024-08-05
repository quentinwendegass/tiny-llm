import logging
import os.path
from importlib import util as import_util
from pathlib import Path

import torch

from helper.warmup_scheduler import WarmupCosineAnnealingScheduler
from model.gpt import GPT


def load_params(name):
    config_path = (
        Path(__file__)
        .resolve()
        .parent.joinpath(f"../../configurations/{name}/configuration.py")
        .resolve()
    )
    module_name = os.path.splitext(os.path.basename(str(config_path)))[0]
    spec = import_util.spec_from_file_location(module_name, str(config_path))
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module_param_dict = module.__dict__
    return {
        "vocab_size": module_param_dict["vocab_size"],
        "model_dim": module_param_dict["model_dim"],
        "num_head": module_param_dict["num_head"],
        "num_hidden": module_param_dict["num_hidden"],
        "num_blocks": module_param_dict["num_blocks"],
        "context_len": module_param_dict["context_len"],
        "dropout": module_param_dict["dropout"],
        "stride": module_param_dict["stride"],
        "batch_size": module_param_dict["batch_size"],
        "warmup": module_param_dict["warmup"],
        "gradient_accumulation_steps": module_param_dict["gradient_accumulation_steps"],
        "learning_rate": module_param_dict["learning_rate"],
        "betas": module_param_dict["betas"],
        "eps": module_param_dict["eps"],
        "validation_freq": module_param_dict["validation_freq"],
        "data_in_memory": module_param_dict["data_in_memory"],
    }


def create_model(device, param_dict):
    return GPT(
        param_dict["vocab_size"],
        param_dict["model_dim"],
        param_dict["num_head"],
        param_dict["num_hidden"],
        param_dict["num_blocks"],
        param_dict["context_len"],
        param_dict["dropout"],
        None,
        device,
    ).to(device)


def create_optimizer(model, params):
    return torch.optim.Adam(
        model.parameters(), betas=params["betas"], eps=params["eps"]
    )


def create_scheduler(optimizer, max_steps, params):
    return WarmupCosineAnnealingScheduler(
        optimizer, max_steps, params["warmup"], params["learning_rate"]
    )


def get_checkpoint_save_path(name):
    return str(
        Path(__file__)
        .resolve()
        .parent.joinpath(f"../../configurations/{name}/checkpoint.pt")
        .resolve()
    )


def load_checkpoint_state(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = create_model(device, checkpoint["params"])
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = create_optimizer(model, checkpoint["params"])
    scheduler = create_scheduler(optimizer, 0, checkpoint["params"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    start_step = checkpoint["step"]

    return model, optimizer, scheduler, start_epoch, start_step, checkpoint["params"]


def create_new_state(params, device, max_steps):
    model = create_model(device, params)
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, max_steps, params)

    start_epoch = 0
    start_step = 0

    return model, optimizer, scheduler, start_epoch, start_step, params


def save_checkpoint(
    name, epoch, step, model, optimizer, scheduler, train_loss, val_loss
):
    param_dict = load_params(name)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "params": param_dict,
        },
        get_checkpoint_save_path(name),
    )
    logging.info(f"New model saved to {get_checkpoint_save_path(name)}")
