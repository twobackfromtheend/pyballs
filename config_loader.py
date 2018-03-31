import configparser


class PresetConfig(configparser.ConfigParser):

    def __init__(self):
        super().__init__()

    def read(self, file_name):
        self.file_name = file_name
        return super().read(file_name)

    def get_presets():
        pass


def load_config(file_name):
    config = PresetConfig()
    config.read(file_name)
    return config
