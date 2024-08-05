import argparse
import configparser
import os
from pathlib import Path

import runpod
from runpod.cli.utils.ssh_cmd import SSHConnection

from helper.helper import get_checkpoint_save_path


def get_config(model_name):
    config = configparser.ConfigParser()
    config.read("runpod.ini")

    parsed_config = dict()
    parsed_config["runpod_api_key"] = config["DEFAULT"].get("ApiKey")
    parsed_config["storage_size"] = config["DEFAULT"].getint("StorageSize", 20)
    parsed_config["use_rclone"] = config["DEFAULT"].getboolean("UseRclone", False)
    parsed_config["gpu"] = config["DEFAULT"].get("Gpu", "NVIDIA A40")
    parsed_config["enable_wandb"] = config["wandb"].getboolean("Enabled", True)
    parsed_config["wandb_api_key"] = config["wandb"]["ApiKey"]
    parsed_config["rclone_config"] = config["rclone"].get("Config")
    parsed_config["rclone_transfers"] = config["rclone"].getint("Transfers", 24)
    parsed_config["rclone_training_src"] = config["rclone"].get("TrainingSrc")

    if parsed_config["enable_wandb"] and not parsed_config["wandb_api_key"]:
        raise Exception("No Wandb API key provided")

    if not parsed_config["runpod_api_key"]:
        raise Exception("No Runpod API key provided")

    if (
        parsed_config["use_rclone"]
        and not parsed_config["rclone_training_src"]
        and not parsed_config["rclone_config"]
    ):
        raise Exception("Missing rclone config")

    if parsed_config["use_rclone"]:
        parsed_config["rclone_training_src"] = parsed_config[
            "rclone_training_src"
        ].replace("{model_name}", model_name)

    return parsed_config


def create_pod(name, gpu, storage_size):
    pod = runpod.create_pod(
        name,
        "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
        gpu,
        volume_in_gb=storage_size,
        volume_mount_path="/workspace",
        ports="8888/http,22/tcp",
        container_disk_in_gb=storage_size,
        cloud_type="SECURE",
    )
    return pod["id"]


def transfer_source_code(ssh):
    ssh.run_commands(["apt-get update --yes", "apt-get install --yes rsync"])
    ssh.rsync(os.path.join(os.getcwd(), "."), "/workspace")


def transfer_with_rclone(ssh, config, transfers, src, dst):
    commands = [
        f"rclone config create {config} --non-interactive",
        (
            f"rclone copy -P --fast-list --checksum --transfers={transfers} remote:{src} {dst} "
            "&> ~/rclone.log; rclone config delete remote"
        ),
    ]

    ssh.run_commands(commands)


def transfer_training_data(ssh: SSHConnection, model_name, config):
    if config["use_rclone"]:
        transfer_with_rclone(
            ssh,
            config["rclone_config"],
            config["rclone_transfers"],
            config["rclone_training_src"],
            f"/workspace/configurations/{model_name}",
        )
    else:
        ssh.put_file(
            f"configurations/{model_name}/tokenized_data.h5",
            f"/workspace/configurations/{model_name}",
        )


def install_dependencies(ssh):
    ssh.run_commands(["cd /workspace && pip install ."])


def write_temp_pod_id(pod_id):
    with open(".tmp_runpod_run", "w") as f:
        f.write(pod_id)


def get_temp_pod_id():
    if not Path(".tmp_runpod_run").exists():
        raise Exception("No pod is currently running")

    with open(".tmp_runpod_run", "r") as f:
        return f.read()


def delete_temp_pod_id():
    os.remove(".tmp_runpod_run")


def configure_runpod(config):
    runpod.api_key = config["runpod_api_key"]


def spin_up_cloud_env(args):
    config = get_config(args.model_name)
    configure_runpod(config)
    pod_id = create_pod(args.model_name, config["gpu"], config["storage_size"])
    try:
        ssh = SSHConnection(pod_id)
        transfer_source_code(ssh)
        transfer_training_data(ssh, args.model_name, config)
        install_dependencies(ssh)
        write_temp_pod_id(pod_id)
    except Exception as e:
        runpod.terminate_pod(pod_id)
        raise e


def start_training(args):
    config = get_config(args.model_name)
    configure_runpod(config)
    pod_id = get_temp_pod_id()
    ssh = SSHConnection(pod_id)

    if config["enable_wandb"]:
        ssh.run_commands([f"wandb login {config["wandb_api_key"]}"])

    start_training_cmd = f"cd /workspace && python src/training.py {args.model_name}"

    if not config["enable_wandb"]:
        start_training_cmd += " --no-wandb"

    if args.epochs:
        start_training_cmd += f" --epochs {args.epochs}"

    ssh.run_commands([start_training_cmd])

    if args.terminate:
        shutdown_pod(args)


def download_checkpoint(args):
    configure_runpod(get_config(""))
    pod_id = get_temp_pod_id()
    ssh = SSHConnection(pod_id)
    ssh.get_file(
        f"/workspace/configurations/{args.model_name}/checkpoint.pt",
        get_checkpoint_save_path(args.model_name),
    )


def connect_to_pod(_):
    configure_runpod(get_config(""))
    pod_id = get_temp_pod_id()
    ssh = SSHConnection(pod_id)
    ssh.launch_terminal()


def shutdown_pod(_):
    configure_runpod(get_config(""))
    pod_id = get_temp_pod_id()
    runpod.terminate_pod(pod_id)
    delete_temp_pod_id()


def main():
    parser = argparse.ArgumentParser(prog="Runpod deployment")
    subparsers = parser.add_subparsers()
    deploy_parser = subparsers.add_parser("deploy")
    connect_parser = subparsers.add_parser("connect")
    shutdown_parser = subparsers.add_parser("shutdown")
    training_parser = subparsers.add_parser("train")
    download_checkpoint_parser = subparsers.add_parser("download-checkpoint")

    deploy_parser.add_argument("model_name", type=str, help="Name of the model")

    training_parser.add_argument("model_name", type=str, help="Name of the model")
    training_parser.add_argument("-e", "--epochs", type=int, help="Number of epochs")
    training_parser.add_argument(
        "--terminate", action=argparse.BooleanOptionalAction, default=True
    )

    download_checkpoint_parser.add_argument(
        "model_name", type=str, help="Name of the model"
    )

    deploy_parser.set_defaults(func=spin_up_cloud_env)
    connect_parser.set_defaults(func=connect_to_pod)
    shutdown_parser.set_defaults(func=shutdown_pod)
    training_parser.set_defaults(func=start_training)
    download_checkpoint_parser.set_defaults(func=download_checkpoint)

    args = parser.parse_args()
    args.func(args)


main()
