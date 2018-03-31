from gui.qt_root import AnalyserWindow
from config_loader import load_config


CONFIG_FILE_NAME = 'presets.ini'


config = load_config(CONFIG_FILE_NAME)
a = AnalyserWindow.main(config)
