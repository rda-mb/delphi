import os

import click
import torch

from delphi.core.config import UserConfig
from delphi.train import infer, setup_data


# #########################    MAIN SCRIPT   ###############################
@click.command(
    context_settings=dict(ignore_unknown_options=True),
    help="Pass in your config file. Any config variable can be overwritten via argument passing. "
    "E.g. python train.py --config '/path/to/config' lr 0.005 batch_size 128",
)
@click.option(
    "-c",
    "--config",
    help="Path to config.yaml file.",
    type=str,
    default=os.getcwd() + r"\darts_config.yaml",
)
def main(config: str) -> None:
    """Main training function of delphi

    Args:
        config (str): Path to user config.yaml file.
        cpu (bool): Whether to use cpu, regardless of GPU availability.
        other_args (tuple): Tuple of (value, key, value, key, ...) to overwrite config params.
    """

    configs = UserConfig(config)
    configs.hparams.stage = "forecast"
    # in case model trained on GPU but inference on CPU
    if not torch.cuda.is_available():
        configs.hparams.device = "cpu"

    # configure data and pipelines
    configs, data, pipes = setup_data(configs)
    target_pipe, _, _ = pipes

    infer(configs, data, target_pipe)


if __name__ == "__main__":
    main()
