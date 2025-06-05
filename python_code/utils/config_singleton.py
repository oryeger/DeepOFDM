import os
from typing import Any
import yaml
from dir_definitions import CONFIG_PATH

class Config:
    __instance = None

    def __new__(cls):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance.config = None
            Config.__instance.reload_config()
        return Config.__instance

    def reload_config(self, config_path=None):
        if config_path is None:
            config_path = CONFIG_PATH
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.config_name = os.path.splitext(os.path.basename(config_path))[0]
        for k, v in config.items():
            setattr(self, k, v)

    def set_value(self, field: Any, value: Any):
        setattr(self, field, value)
