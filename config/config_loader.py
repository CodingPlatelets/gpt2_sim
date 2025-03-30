import json
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_path=None):
        self.config_path = config_path or Path(
            __file__).parent / "gpt2_config.json"
        self._config = self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return json.load(f)

    @property
    def memory_config(self):
        return self._config.get("memory", {})

    @property
    def hardware_config(self):
        return self._config.get("hardware", {})

    def get_memory_map(self):
        return self.memory_config.get("map", {})
