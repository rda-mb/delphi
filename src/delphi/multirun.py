from delphi.core.config import UserConfig
from delphi.train import train


# either:
#  - generate 1 config file, save it as tempfile and call train(['-c', config_file])
#  - use 1 generic config file, and call train(['-c', config_file, '--target={target}' etc])
# define dict of target: datasources
segments = ["Aframax", "LR1", "LR2", "MR", "Suezmax", "VLCC"]
config_file = (
    r"C:\Users\admrda\OneDrive - Maersk Broker\python\TC handover\darts_tankerTC_configs.yaml"
)
configs = UserConfig(config_file)
# overwrite params in configs.hparams.variable
# kwargs = {'target': t, }
# configs.overwrite_hparams_from_dict(kwargs)
train(configs)
