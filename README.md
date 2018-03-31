# PyBAllS
The Python Basic Animated coLLision Simulator.

## Features
- GUI
    - Fulfil all your simulation needs without touching code!
    - Get to click buttons
- 3.2 Simulation Methods
    1. Recalculate collision times for all balls after every collision.
    2. Recalculate collision times for required balls only after each collision.
    3. Calculate overlapping balls at intervals

## Usage
#### Basics
1. Run `python start.py`.
2. Click buttons.
3. *(Optional)* Open presets.ini in your favourite text editor (Notepad++ works very well for this)

#### Not-So-Basic Basics
- Creation
    - This tab allows for creating and editing gas, simulation and animation presets
    - To create a preset, modify the preset name such that it does not match any existing preset. The 'Save Preset' button should now read 'Create Preset'. Click it to create a new preset with the parameters visible.
    - Modifying parameters does NOT modify the preset. To modify the preset, first select the preset, modify the parameters, then click 'Save Preset'.
    - The idea behind this behaviour is to allow easy creation of similar presets (i.e. click preset, change a parameter, add a '1' behind the preset name, 'Create Preset'!)
    - Gas
        - Click a gas to load it, click 'Save Preset' to save it.
        - If mass is 0 for both, the program generates a sensible mass with a fixed density.
    - Simulation
        - Select an Existing Preset, and the gases in this preset are selected in the Gases ListWidget to its right (NOT the Gas tab) (and gases not in the preset are deselected)
        - Select/deselect gases by clicking on them.
        - To add/remove a gas from an existing simulation preset, select the simulation preset, select/deselect the gas if you want to add/remove it, then click 'Save Preset'
    - Animation
        - If you've gotten this far, this should be simple.
        - Speed up simply increases the duration simulated per frame displayed. This does not affect stats, but simply hastens animation. Useful for data collection while watching pretty pictures.

#### Suggested playthrough:
###### Preamble
- Insert `{'duration_limit': 5}` in Overrides to avoid having to wait 30s to see results.
- Move the velocity plot with the pan tool (in the nav toolbar) to update it. It's updated whenever stats are printed if there's an animation. You can click the home button (also in the nav toolbar) after having moved it to update the plot. The plot can also be rescaled with the zoom tool, or with the configure subplots thing (both in the nav toolbar)
- If there's no animation, wait till it's done. Or kill it and restart it with more sensible values.
- Feel free to experiment with the various simulation methods, either as an override, or by editing presets.

###### SIM, ANIM:
1. DEFAULTSIM, DEFAULTANIM
    - Look at balls go
2. SIM2, DEFAULTANIM
    - Look at more balls go
    - but with lag
3. SIM2, ANIM_DOUBLE
    - Look at the same balls go faster
    - but with roughly equal lag - indicates matplotlib is the culprit, not the simulation method
4. SIM3D, DEFAULTANIM
    - Look at the balls go IN 3D
5. SIM2GAS, DEFAULTANIM
    - Do not miss out on the velocity distributions of this one. It's beautiful.
6. SIM3D, DEFAULTANIM
    - Just more small balls in 3D
    - Feel free to skip this one.
7. RWSimN250, NONE
    - Short for Real-World Simulation of N2, 50 particles.
    - Do not try to animate this unless you want blood on your hands.
8. RWSim50Air, NONE
    - Same as above, just that it's 39 N2, 11 O2 (Just like the air around you!)
    - Animate at your own risk. Also not for pacifist playthroughs.
9. Play around with TESTSIM, DEFAULTANIM, modification, creation of new simulations.
10. Uncomment a `s = Sim...` and the `s.run()` line. Then run `python balls.py`. Especially try out the simulations with `enumerate` in their gas parameters. They're particularly fun to watch.

#### Advanced
The 'Additional Investigations' menubar item provides a window to a whole new thermodynamic world. The parameters can only be modified in the presets.ini file.
These give the plots seen in the report.
The data generated here is saved in sim_stats.csv for troubleshooting purposes. Or for future plotting. It's a good idea to keep it anyway, as it can take quite a while to generate.

## Notes
- The included requirements.txt was made using a Python 3.6.3 anaconda virtual env using pip freeze. Think of the requirements indicated as sufficient but not necessary.
- Effort was made to restrict the libraries used to those found in a default Spyder installation. This meant that I could not use tqdm to give nice progress bars and an estimation of duration remaining for certain data-collecting functions.
- Simulation method 2 is not as good as it sounds. See report for details.
- Report included in zip as it's referred to here.

## WIP
- The ArraySimulation will likely provide huge gains in simulation efficiency
- More thermodynamics plotting stuff can be done. In particular, 2D colour plots for stuff look really nice. These can be implemented as quick, less accurate plots. E.g. Pressure in colour for given temperature and volume (on x and y axes). Colour can also show non-ideal gas behaviour as volume decreases (e.g. non-linear line of colour change)