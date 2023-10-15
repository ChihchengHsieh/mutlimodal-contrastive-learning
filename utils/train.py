from copy import deepcopy
import math
import sys
import time
from .loggers import MetricLogger, SmoothedValue
import numpy as np
import torch.utils.data as data
import torch
import datetime

from datetime import datetime
from config import ConfigArgs
from .tensor import nested_to_device
from .dict import reduce_dict
from .checkpoint import get_model_path, save_checkpoint, remove_existing_cp


class TrainingTimer(object):
    def __init__(self) -> None:
        self.init_t = datetime.now()
        self.start_t = None
        self.end_t = None
        self.last_epoch = None
        self.epoch_start_t = None

    def start_training(
        self,
    ):
        self.start_t = datetime.now()

    def start_epoch(
        self,
    ):
        self.epoch_start_t = datetime.now()

    def end_epoch(self, epoch):
        self.last_epoch = epoch

        finish_time = datetime.now()
        epoch_took = (finish_time - self.epoch_start_t).total_seconds()
        sec_already_took = (finish_time - self.start_t).total_seconds()
        speed = sec_already_took / self.last_epoch

        return epoch_took, sec_already_took, speed

    def end_training(
        self,
    ):
        self.end_t = datetime.now()

    def has_took_sec_from_init(
        self,
    ):
        return (datetime.now() - self.init_t).total_seconds()

    def has_took_sec(
        self,
    ):
        return (datetime.now() - self.start_t).total_seconds()


class TrainingInfo:
    def __init__(self, config: ConfigArgs):
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf
        self.best_val_loss_model_path = None
        self.final_model_path = None
        self.config = config
        self.timer = TrainingTimer()

        self.epoch = 0
        super(TrainingInfo).__init__()

    def __str__(self):
        title = "=" * 40 + \
            f"For Training [{self.config.training.name} - {self.config.model.name}]" + "=" * 40
        section_divider = len(title) * "="

        return (
            title + "\n" + str(self.config) + "\n" + section_divider + "\n\n"
            f"Best model has been saved to: [{self.best_val_loss_model_path}]"
            + "\n"
            f"The final model has been saved to: [{self.final_model_path}]"
            + "\n\n"
            + section_divider
        )


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def epoch_end_print(train_info: TrainingInfo, early_stopper: EarlyStopper, num_epochs):
    epoch_took, sec_already_took, speed = train_info.timer.end_epoch(
        train_info.epoch
    )
    print_str = f"| Epoch [{train_info.epoch}] Done | It has took [{sec_already_took/60:.2f}] min, Avg time: [{speed:.2f}] sec/epoch | Estimate time for [{num_epochs}] epochs: [{speed*num_epochs/60:.2f}] min | Epoch took [{epoch_took}] sec | "
    if early_stopper:
        print_str += f" Patience [{early_stopper.counter}] |"
    print(print_str)


def get_datasets(
    dataset_args: dict,
    dataset_class: data.Dataset,
):

    train_dataset = dataset_class(
        **dataset_args,
        split_str="train",
    )

    val_dataset = dataset_class(
        **dataset_args,
        split_str="val",
    )

    test_dataset = dataset_class(
        **dataset_args,
        split_str="test",
    )

    return train_dataset, val_dataset, test_dataset


def __collate_fn(batch):
    return tuple(zip(*batch))


def __get_dataloader_g(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_dataloaders(
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    test_dataset: data.Dataset = None,
    batch_size: int = 4,
    seed: int = 0,
    drop_last: bool = False,
):

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=__collate_fn,
        generator=__get_dataloader_g(seed),
        drop_last=drop_last,
    )

    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=__collate_fn,
        generator=__get_dataloader_g(seed),
        drop_last=drop_last,
    )

    if test_dataset:
        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=__collate_fn,
            generator=__get_dataloader_g(seed),
            drop_last=drop_last,  # if we don't make it == True, we may get a batch with only size=1
        )
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader


def print_training_status(train_info, train_infos, config):
    trained_model_prt = ("=" * 30) + "Trained Models" + ("=" * 30)
    print(trained_model_prt)
    for _t in train_infos:
        print(_t.best_performance_model_path)
    print(("=" * len(trained_model_prt)))

    print(f"Training model: [{config.model.name}]")
    print(train_info)


def train_one_epoch(model, optimiser, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for inputs, targets in metric_logger.log_every(data_loader, print_freq, header):
        inputs = torch.stack(nested_to_device(inputs, device), axis=0)
        targets = nested_to_device(targets, device)
        loss_dict, _ = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimiser.zero_grad()
        losses.backward()
        optimiser.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimiser.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device, evaluator=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    # another evaluator

    for inputs, targets in metric_logger.log_every(data_loader, 100, header):
        inputs = torch.stack(nested_to_device(inputs, device), axis=0)
        targets = nested_to_device(targets, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        loss_dict, outputs = model(inputs, targets)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        outputs = outputs.to(cpu_device)
        targets = nested_to_device(targets, cpu_device)
        model_time = time.time() - model_time

        if evaluator:
            evaluator.update(
                outputs,
                targets,
            )

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time,
                             loss=losses_reduced,
                             **loss_dict_reduced,
                             )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.set_num_threads(n_threads)

    if evaluator:
        return metric_logger, evaluator

    return metric_logger


def check_best(
    train_info,
    model,
    optimiser,
    val_evaluator,
) -> TrainingInfo:
    # Targeting the model with higher Average Recall and Average Precision.
    if train_info.val_losses[-1]['loss'] < train_info.best_val_loss:
        # do evaluation on test.
        previous_best_model = deepcopy(train_info.best_val_loss_model_path)
        model_path = get_model_path(
            train_info, val_evaluator.get_performance())
        train_info.best_val_loss_model_path = model_path
        train_info.final_model_path = model_path
        train_info = save_checkpoint(
            train_info=train_info,
            model=model,
            model_path=model_path,
            optimiser=optimiser,
        )
        train_info.best_val_loss_model_path = train_info.final_model_path
        train_info.best_val_loss = train_info.val_losses[-1]['loss']
        if previous_best_model:
            remove_existing_cp(previous_best_model)

    return train_info


def end_train(
    train_info,
    model,
    optimiser,
    val_evaluator,
) -> TrainingInfo:

    train_info.timer.end_training()
    sec_took = train_info.timer.has_took_sec()
    print(
        f"| Training Done, start testing! | [{train_info.epoch}] Epochs Training time: [{sec_took}] seconds, Avg time / Epoch: [{sec_took/train_info.epoch}] seconds"
    )

    model_path = get_model_path(train_info, val_evaluator.get_performance())
    train_info.final_model_path = model_path
    train_info = save_checkpoint(
        train_info=train_info,
        model=model,
        model_path=model_path,
        optimiser=optimiser,
    )

    print(train_info)

    return train_info


def set_weights_trainable(model, optimiser, model_part = None):
    for n, param in model.named_parameters():
        if (model_part is None or n.startswith(model_part)) and param.requires_grad == False:
            param.requires_grad = True
            optimiser.add_param_group({'params': param})
    return model, optimiser