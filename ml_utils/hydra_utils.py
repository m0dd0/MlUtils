from pathlib import Path
from typing import List

import hydra
from omegaconf import OmegaConf


class PartialModule:
    def __init__(self, _real_target_, **module_kwargs):
        self._real_target_ = _real_target_
        self.module_kwargs = module_kwargs

    def instantiate(self, **missing_kwargs):
        all_kwargs = {**self.module_kwargs, **missing_kwargs}

        if isinstance(self._real_target_, str):
            return hydra.utils.instantiate(
                {"_target_": self._real_target_, **all_kwargs}
            )

        return self._real_target_(**all_kwargs)


class ConfigSaver:
    def __init__(self, target_path: Path):
        self.target_path = target_path

    def __call__(self, config_keys: List[str] = None):
        config_keys = config_keys or []

        # assumes that hydra is already initialized
        cfg = hydra.compose(config_name="config")
        for key in config_keys:
            cfg = cfg[key]

        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.target_path, "w") as f:
            OmegaConf.save(OmegaConf.to_container(cfg), f)
