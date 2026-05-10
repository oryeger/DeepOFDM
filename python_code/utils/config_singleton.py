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

        # The make_64QAM_16QAM_percentage=50 three-way split branch (in both
        # mimo_channel_dataset.py and evaluate.py) only fires when mod_pilot==64.
        # Force it on so the flag isn't silently ignored when mod_pilot is left at -1.
        if getattr(self, 'make_64QAM_16QAM_percentage', 0) == 50 and getattr(self, 'mod_pilot', -1) != 64:
            warned = getattr(Config, '_warned_mod_pilot_override', False)
            if not warned:
                print(f"[config] make_64QAM_16QAM_percentage=50 requires mod_pilot=64; "
                      f"overriding mod_pilot={getattr(self, 'mod_pilot', -1)} -> 64")
                Config._warned_mod_pilot_override = True
            self.mod_pilot = 64

        # The percentage=50 curriculum and increase_prime_modulation are two
        # different mod-mismatch pathways and must not be combined.
        if getattr(self, 'make_64QAM_16QAM_percentage', 0) == 50 and getattr(self, 'increase_prime_modulation', False):
            raise ValueError(
                "make_64QAM_16QAM_percentage=50 cannot be used with increase_prime_modulation=True; "
                "set increase_prime_modulation=False or change the percentage."
            )

        model = getattr(self, 'channel_model', 'N')
        self.delay_spread = {'A': 30e-9, 'B': 100e-9}.get(model[0], 300e-9)

    def set_value(self, field: Any, value: Any):
        setattr(self, field, value)
