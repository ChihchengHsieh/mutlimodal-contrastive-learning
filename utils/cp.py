import os, torch, pickle

from torch import nn
from torch.optim.optimizer import Optimizer


def save_checkpoint(
    model_path: str,
    training_args,
    model_args,
    model: nn.Module,
    optimizer: Optimizer = None,
):
    saving_dict = {"model": model.state_dict()}

    if optimizer:
        saving_dict["optimizer"] = optimizer.state_dict()

    saving_folder = os.path.join("checkpoints", model_path)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(saving_folder, exist_ok=True)

    torch.save(
        saving_dict,
        os.path.join("checkpoints", model_path, "model"),
    )

    # saving the model_args.
    with open(
        os.path.join("checkpoints", model_path, "training_args.pkl"),
        "wb",
    ) as training_f:
        pickle.dump(training_args, training_f)

    # saving the model_args.
    with open(
        os.path.join("checkpoints", model_path, "model_args.pkl"),
        "wb",
    ) as model_f:
        pickle.dump(model_args, model_f)

    return model_path


def load_checkpoint(model_path, device):
    folder_path = os.path.join("checkpoints", model_path)

    with open(os.path.join(folder_path, "training_args.pkl"), "rb") as f:
        training_args = pickle.load(f)

    with open(os.path.join(folder_path, "model_args.pkl"), "rb") as f:
        model_args = pickle.load(f)

    cp = torch.load(os.path.join(folder_path, "model"), map_location=device)

    return training_args, model_args, cp
