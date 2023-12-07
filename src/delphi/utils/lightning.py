import warnings

import optuna
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch


def format_pl_trainer_params(
    params: dict, model_name: str, device: str, reset_model: bool = True
) -> dict:
    """Formatting input parameters for TFTModel related to the trainer setup.

    Args:
        params (dict): hyper parameters from config.yaml file.
        model_name (str): Model name as saved in darts_logs/{model_name}.
        device (str): Device used for training, cpu or cuda.
        reset_model (bool): Overwrite existing model with the same name. Default True.

    Returns:
        dict: Dictionnary containing all necessary keys for model input.
    """
    # callbacks
    callbacks = []
    if "early_stopping" in params and params["early_stopping"]:
        early_stop = EarlyStopping(**params["early_stopping"])
        callbacks.append(early_stop)

    tr_params = {
        "n_epochs": params["n_epochs"],
        "batch_size": params["batch_size"],
        "optimizer_kwargs": {"lr": params["learning_rate"]},
        "optimizer_cls": getattr(torch.optim, params["optimizer"]),
        "pl_trainer_kwargs": {"callbacks": callbacks},
        "model_name": model_name,
        "force_reset": reset_model,
        "save_checkpoints": True,
    }

    # learning rate scheduler
    if "lr_scheduler" in params and params["lr_scheduler"]:
        print("Using learning rate scheduler: ", params["lr_scheduler"])
        tr_params["lr_scheduler_cls"] = getattr(torch.optim.lr_scheduler, params["lr_scheduler"])

    if "cuda" in device:
        print(f"Setting up GPU training on device {device}.")
        tr_params["pl_trainer_kwargs"]["accelerator"] = "gpu"
        tr_params["pl_trainer_kwargs"]["devices"] = -1
    else:
        tr_params["pl_trainer_kwargs"]["accelerator"] = "cpu"

    return tr_params


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/main/pytorch
    /pytorch_lightning_simple.py>` if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names
            thus depend on how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
