import os
import yaml


class ConfigParser:
    # YMAL parser for configs
    def __init__(self, config):
        self._config = {}
        self.reset_config()
        self.parse_config(config)

    def parse_config(self, file):
        with open(file, 'r') as fid:
            yaml_config = yaml.safe_load(fid)
        self.parse_dict(yaml_config)

    def parse_dict(self, input_dict, parent=None):
        if parent is None:
            parent = self._config
        for key, val in input_dict.items():
            if isinstance(val, dict):
                if key not in parent.keys():
                    parent[key] = {}
                self.parse_dict(val, parent[key])
            else:
                parent[key] = val

    def reset_config(self):
        self._config['data'] = {}
        self._config['exper'] = {}
        self._config['model'] = {}
        self._config['eval'] = {}
        self._config['utils'] = {}

    def save_config(self, path_dir, file_name):
        with open(os.path.join(path_dir, file_name), 'w') as fid:
            yaml.dump(self._config, fid)

    @staticmethod
    def save_config_dict(config_dict, path_dir, file_name):
        with open(os.path.join(path_dir, file_name), 'w') as fid:
            yaml.dump(config_dict, fid)

    def update(self, config):
        self.reset_config()
        self.parse_config(config)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def merge_configs(self, path_models):
        # parse training config
        with open(path_models, 'r') as fid:
            config = yaml.safe_load(fid)
        # overwrite with config settings
        self.parse_dict(config, self._config)
        return config
