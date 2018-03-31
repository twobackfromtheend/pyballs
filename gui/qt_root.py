from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget
from PyQt5 import QtCore
from gui.qt_gui import Ui_MainWindow
import sys
import math
import balls
import ast
import pprint
import investigations as inv


class AnalyserWindow(QMainWindow, Ui_MainWindow):
    """Creates Qt Window linked to functions"""

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # create tracking lists
        self.gas_presets = []
        self.sim_presets = []
        self.anim_presets = []

        self.create_functions()

    def main(config):
        app = QApplication(sys.argv)
        window = AnalyserWindow()
        window.load_presets_from_config(config, statusbar_update=False)
        window.show()
        app.exec_()

    def create_functions(self):
        # menubar
        self.actionReload.triggered.connect(self.load_presets_from_config)
        self.actionShow_Gas_Laws.triggered.connect(self.run_show_gas_laws)
        self.actionPlot_B_against_V.triggered.connect(self.run_plot_b_against_v)
        self.actionFind_B_Graphically.triggered.connect(self.run_find_b)

        # list widgets
        self.gas_list_widget.itemClicked.connect(self.load_preset_from_list)
        self.sim_list_widget.itemClicked.connect(self.load_preset_from_list)
        self.anim_list_widget.itemClicked.connect(self.load_preset_from_list)

        # link create preset button to lineedit (if preset name exists, change
        # text to 'save preset', else 'create preset')
        self.gas_preset_edit.textChanged.connect(self.preset_name_edit)
        self.sim_preset_edit.textChanged.connect(self.preset_name_edit)
        self.anim_preset_edit.textChanged.connect(self.preset_name_edit)

        # create preset buttons
        self.gas_create_button.clicked.connect(self.create_preset)
        self.sim_create_button.clicked.connect(self.create_preset)
        self.anim_create_button.clicked.connect(self.create_preset)

        self.run_sim_button.clicked.connect(self.run_sim)

    def create_preset(self, *args):
        sending_button = self.sender()
        if sending_button is self.gas_create_button:
            # create gas preset
            preset_name = self.gas_preset_edit.text()
            preset_parameters = {
                'type': 'gas',
                # preserves e notation
                'most_probable_velocity': str(self.gas_vel_box.value()) + 'e' + str(self.gas_vele_box.value()),
                'mass': str(self.gas_mass_box.value()) + 'e' + str(self.gas_masse_box.value()),
                'radius': str(self.gas_rad_box.value()) + 'e' + str(self.gas_rade_box.value()),
                'count': str(self.gas_count_box.value())
            }

            self.config[preset_name] = preset_parameters
            self.statusbar.showMessage(
                'Saved gas preset: %s' % preset_name, 10000)
        if sending_button is self.sim_create_button:
            # create sim preset
            preset_name = self.sim_preset_edit.text()
            selected_gases = [gas_list_widget_item.text(
            ) for gas_list_widget_item in self.sim_gas_list_widget.selectedItems()]
            preset_parameters = {
                'type': 'sim',
                'gases': ', '.join(map(str, selected_gases)),
                # preserves e notation
                'container_radius': str(self.sim_rad_box.value()) + 'e' + str(self.sim_rade_box.value()),
                'simulation_method': str(self.sim_method_box.currentText()),
                'dimensions': str(self.sim_dim_box.value())
            }

            self.config[preset_name] = preset_parameters
            self.statusbar.showMessage(
                'Saved sim preset: %s' % preset_name, 10000)

        if sending_button is self.anim_create_button:
            # create anim preset
            self.anim_presets.append('hi')
            print(self.anim_presets)
            self.statusbar.showMessage(
                'Saved anim preset: %s' % preset_name, 10000)

        # save presets into file
        with open(self.config.file_name, 'w') as config_file:
            self.config.write(config_file)
        # reload presets from config
        self.load_presets_from_config(statusbar_update=False)

    def load_presets_from_config(self, config=None, statusbar_update=True):
        # reloads all presets from config
        if config:
            self.config = config
        else:
            config = self.config

        # reset existing lists
        self.gas_presets, self.sim_presets, self.anim_presets = [], [], []
        ignored_presets = []
        for _preset_name in config.sections():
            _preset_type = config[_preset_name].get('type')
            if _preset_type is None:
                ignored_presets.append(_preset_name)

            if _preset_type == 'gas':
                self.gas_presets.append(_preset_name)
            elif _preset_type == 'sim':
                self.sim_presets.append(_preset_name)
            elif _preset_type == 'anim':
                self.anim_presets.append(_preset_name)

        self.gas_list_widget.clear()
        self.gas_list_widget.addItems(self.gas_presets)
        self.sim_list_widget.clear()
        self.sim_list_widget.addItems(self.sim_presets)
        self.sim_gas_list_widget.clear()  # list widget of gases to add to simulation
        self.sim_gas_list_widget.addItems(self.gas_presets)
        self.anim_list_widget.clear()
        self.anim_list_widget.addItems(self.anim_presets)

        self.simulation_combobox.clear()
        self.simulation_combobox.addItems(self.sim_presets)
        self.anim_presets.append('None')
        self.animation_combobox.clear()
        self.animation_combobox.addItems(self.anim_presets)

        if statusbar_update:
            self.statusbar.showMessage('Reloaded presets.')

    def load_preset_from_list(self):
        sending_widget = self.sender()

        if 'gas' in sending_widget.objectName():
            # load gas preset
            _preset_name = sending_widget.currentItem().text()
            print('Selected %s preset in %s. Loading preset.' %
                  (_preset_name, sending_widget.objectName()))
            _preset = self.config[_preset_name]

            _vel = _preset.getfloat('most_probable_velocity')
            _vel_exponent = math.floor(math.log10(_vel))
            # aka mantissa or coefficient, is the part before the exponent
            _vel_significand = _vel / 10.**_vel_exponent
            self.gas_vel_box.setValue(_vel_significand)
            self.gas_vele_box.setValue(_vel_exponent)

            _mass = _preset.getfloat('mass')
            if _mass == 0:
                self.gas_mass_box.setValue(0)
                self.gas_masse_box.setValue(0)
            else:
                _mass_exponent = math.floor(math.log10(_mass))
                _mass_significand = _mass / 10.**_mass_exponent
                self.gas_mass_box.setValue(_mass_significand)
                self.gas_masse_box.setValue(_mass_exponent)

            _rad = _preset.getfloat('radius')
            _rad_exponent = math.floor(math.log10(_rad))
            _rad_significand = _rad / 10.**_rad_exponent
            self.gas_rad_box.setValue(_rad_significand)
            self.gas_rade_box.setValue(_rad_exponent)

            _count = _preset.getint('count')
            self.gas_count_box.setValue(_count)

            self.gas_preset_edit.setText(_preset_name)

            self.statusbar.showMessage(
                'Loaded gas preset %s.' % _preset_name, 10000)

        elif 'sim' in sending_widget.objectName():
            # load sim preset
            _preset_name = sending_widget.currentItem().text()
            print('Selected %s preset in %s. Loading preset.' %
                  (_preset_name, sending_widget.objectName()))
            _preset = self.config[_preset_name]

            _rad = _preset.getfloat('container_radius')
            _rad_exponent = math.floor(math.log10(_rad))
            # aka mantissa or coefficient, is the part before the exponent
            _rad_significand = _rad / 10.**_rad_exponent
            self.sim_rad_box.setValue(_rad_significand)
            self.sim_rade_box.setValue(_rad_exponent)

            self.sim_dim_box.setValue(_preset.getint('dimensions'))

            _sim_method = str(_preset.getint('simulation_method'))
            index = self.sim_method_box.findText(
                _sim_method, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.sim_method_box.setCurrentIndex(index)

            # load gases selected for simulation
            sim_gases = [gas_name.strip()
                         for gas_name in _preset.get('gases').split(',')]

            for i in range(self.sim_gas_list_widget.count()):
                gas_list_widget_item = self.sim_gas_list_widget.item(i)
                if str(gas_list_widget_item.text()) in sim_gases:
                    gas_list_widget_item.setSelected(True)
                else:
                    gas_list_widget_item.setSelected(False)

            self.sim_preset_edit.setText(_preset_name)

            self.statusbar.showMessage(
                'Loaded simulation preset %s.' % _preset_name, 10000)

        elif 'anim' in sending_widget.objectName():
            # load anim preset
            _preset_name = sending_widget.currentItem().text()
            print('Selected %s preset in %s. Loading preset.' %
                  (_preset_name, sending_widget.objectName()))
            _preset = self.config[_preset_name]

            _speedup = _preset.getfloat('speed_up')
            self.anim_speed_box.setValue(_speedup)

            _fps = _preset.getint('fps')
            self.anim_fps_box.setValue(_fps)

            _count = _preset.getint('count')
            self.gas_count_box.setValue(_count)

            self.gas_preset_edit.setText(_preset_name)

            self.statusbar.showMessage(
                'Loaded gas preset %s.' % _preset_name, 10000)

    def preset_name_edit(self):
        sending_widget = self.sender()
        _preset_name = sending_widget.text()
        if 'gas' in sending_widget.objectName():
            # check if gas exists
            if _preset_name.lower() in [preset_name.lower() for preset_name in self.gas_presets]:
                self.gas_create_button.setText('Save Preset')
            else:
                self.gas_create_button.setText('Create Preset')

        elif 'sim' in sending_widget.objectName():
            # check if gas exists
            if _preset_name.lower() in [preset_name.lower() for preset_name in self.sim_presets]:
                self.sim_create_button.setText('Save Preset')
            else:
                self.sim_create_button.setText('Create Preset')

        if 'anim' in sending_widget.objectName():
            # check if gas exists
            if _preset_name.lower() in [preset_name.lower() for preset_name in self.anim_presets]:
                self.anim_create_button.setText('Save Preset')
            else:
                self.anim_create_button.setText('Create Preset')

    def run_sim(self):
        sim_preset = self.config[self.simulation_combobox.currentText()]
        anim_preset_name = self.animation_combobox.currentText()
        if anim_preset_name == 'None':
            animation = None
        else:
            animation = True
            anim_preset = self.config[anim_preset_name]

        # get gases
        sim_gases = [gas_name.strip()
                     for gas_name in sim_preset.get('gases').split(',')]
        gases = []
        for _gas_name in sim_gases:
            _gas_preset = self.config[_gas_name]
            mass = _gas_preset.getfloat('mass', None)
            if mass == 0:
                mass = None  # autogenerate mass
            _gas_parameters = {
                'radius': _gas_preset.getfloat('radius'),
                'mass': mass,
                'v_p': _gas_preset.getfloat('most_probable_velocity'),
                'count': _gas_preset.getint('count')
            }
            gases.append(_gas_parameters)

        container_radius = sim_preset.getfloat('container_radius')
        dimensions = sim_preset.getint('dimensions')
        simulation_method = sim_preset.getint('simulation_method')

        if animation:
            animation_presets = {
                'fps': anim_preset.getint('fps'),
                'speed_up': anim_preset.getfloat('speed_up')}
        else:
            animation_presets = {}

        override_text = self.plainTextEdit.toPlainText()
        if override_text:
            # parse override_text
            try:
                overrides = ast.literal_eval(override_text)
            except:
                print('Ignored overrides. Check format. (see ast.literal_eval)')
                overrides = {}
        else:
            overrides = {}
        # print(animation_presets)
        _simulation_parameters = {'gases': gases,
                                  'container_radius': container_radius,
                                  'dimensions': dimensions,
                                  'simulation_method': simulation_method,
                                  'animation': animation}
        # add animation_presets
        for _key, _value in animation_presets.items():
            _simulation_parameters[_key] = _value

        # apply overrides
        for _key, _value in overrides.items():
            _simulation_parameters[_key] = _value

        print('Simulation parameters:')
        pprint.pprint(_simulation_parameters)

        try:
            s = balls.Simulation(**_simulation_parameters)
            print('Simulation generated. Running simulation...')
            s.run()
        except Exception as e:
            # try except and show error message in statusbar (traceback
            # available in console)
            import traceback
            traceback.print_exc()
            self.statusbar.showMessage(str(e))

    def run_show_gas_laws(self):
        _preset = self.config['SHOW_GAS_LAWS']
        duration_limit = _preset.getfloat('duration_limit')
        repeats = _preset.getint('repeats')
        velocity_seed = _preset.getint('velocity_seed')
        points = _preset.getint('points')
        boyle = _preset.getboolean('boyle')
        charles = _preset.getboolean('charles')
        gay = _preset.getboolean('gay')
        inv.show_gas_laws(duration_limit=duration_limit, repeats=repeats, velocity_seed=velocity_seed, points=points, boyle=boyle, charles=charles, gay=gay)

    def run_plot_b_against_v(self):
        _preset = self.config['PLOT_B_AGAINST_VG']
        duration_limit = _preset.getfloat('duration_limit')
        repeats = _preset.getint('repeats')
        velocity_seed = _preset.getint('velocity_seed')
        inv.plot_b_against_V_g(duration_limit=duration_limit, repeats=repeats, velocity_seed=velocity_seed)

    def run_find_b(self):
        _preset = self.config['FIND_B_GRAPHICALLY']
        duration_limit = _preset.getfloat('duration_limit')
        repeats = _preset.getint('repeats')
        velocity_seed = _preset.getint('velocity_seed')
        inv.find_b_graphically(duration_limit=duration_limit, repeats=repeats, velocity_seed=velocity_seed)


