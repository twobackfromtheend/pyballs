from balls import *
import numpy as np
import csv
import os
import pprint
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'


def show_gas_laws(duration_limit=5, repeats=5, velocity_seed=1, points=15, boyle=True, charles=True, gay=True):
    """Plot and show gas laws (boyle, charles, gay). Plot is laid out for all 3"""

    fig = plt.figure(figsize=(15, 9))

    # Base parameters
    base_gas = {'count': 20, 'radius': 1.55e-10,
                'v_p': 4.09e2, 'mass': 4.50e-26}
    base_simulation_parameters = {
        'gases': [base_gas],
        'dimensions': 3,
        'simulation_method': 1,
        'animation': False,
        'container_radius': 5.56e-9,
        'velocity_seed': velocity_seed
    }
    base_volume = 4 / 3 * np.pi * \
        base_simulation_parameters['container_radius']**3

    volumes = np.linspace(0.2, 4, points) * base_volume
    radii = np.cbrt(3 / 4 / np.pi * volumes)

    if charles:
        print("Plotting Charles's Law")
        # Charles's Law
        # V ~ T at constant Pressure
        # use known V, T, label points with pressure
        volumes = volumes  # use above volume
        radii = radii

        V_data = []
        T_data = []
        P_data = []  # to be used as labels
        P_err_data = []

        for _r, _v in zip(radii, volumes):
            multiplier = _v / base_volume
            # generate new simulation with new V
            gas = base_gas.copy()
            gas['v_p'] = gas['v_p'] * np.sqrt(multiplier)  # T = T(v^2)
            _simulation_parameters = base_simulation_parameters.copy()
            _simulation_parameters['gases'] = [gas]
            _simulation_parameters['container_radius'] = _r

            returned_data = find_stats_for_simulation_parameters(
                _simulation_parameters, duration_limit=duration_limit, repeats=repeats, combine_data=True)
            P = np.mean(returned_data['pressure'])
            P_err = np.std(returned_data['pressure'])

            T = np.mean(returned_data['temperature'])  # constant
            V_data.append(_v)
            T_data.append(T)
            P_data.append(P)
            P_err_data.append(P_err)

        # Plot 1: V against T, P labelled
        ax1 = fig.add_subplot(221)
        ax1.plot(V_data, T_data, 'b.')
        ax1.set_title("Charles's Law: $V \propto T$, with P labelled")
        ax1.set_xlabel(r'$V$')
        ax1.set_ylabel(r'$T$')
        ax1.set_ylim(ymin=0)
        ax1.set_xlim(xmin=0)

        labels = [r'$P =$%.2e' % P_data[i] for i in range(points)]  # error in other graph
        for label, x, y in zip(labels, V_data, T_data):
            plt.annotate(label, xy=(x, y), xytext=(-10, 5),
                         textcoords='offset points', ha='center', va='bottom', size='x-small')

        # Plot 2: P against V
        ax2 = fig.add_subplot(222)
        ax2.plot(V_data, P_data, 'b.')
        ax2.errorbar(V_data, P_data, yerr=P_err_data, fmt='none')

        labels = [r'%.2e$\pm$%1.e' % (P_data[i], P_err_data[i]) for i in range(points)]
        for label, x, y in zip(labels, V_data, P_data):
            plt.annotate(label, xy=(x, y), xytext=(5, 5),
                         textcoords='offset points', ha='center', va='bottom', size='x-small')

        ax2.set_title("Charles's Law (2)")
        ax2.set_xlabel(r'$V$')
        ax2.set_ylabel(r'$P$')
        # ax2.set_ylim(ymin=0)
        ax2.set_xlim(xmin=0)

    if boyle:
        print("Plotting Boyle's Law")

        # Boyle's law
        # PV = k at constant T

        # plot p against v, plot k/v line

        P_data = []
        V_data = []
        P_err_data = []

        for _r, _v in zip(radii, volumes):
            _simulation_parameters = base_simulation_parameters.copy()
            _simulation_parameters['container_radius'] = _r
            returned_data = find_stats_for_simulation_parameters(
                _simulation_parameters, duration_limit=duration_limit, repeats=repeats, combine_data=True)
            # pprint.pprint(returned_data)

            P = np.mean(returned_data['pressure'])
            P_err = np.std(returned_data['pressure'])

            P_data.append(P)
            V_data.append(_v)
            P_err_data.append(P_err)

        P_data = np.array(P_data)
        V_data = np.array(V_data)

        ax3 = fig.add_subplot(223)
        ax3.plot(V_data, P_data, 'b.')
        ax3.errorbar(V_data, P_data, yerr=P_err_data, fmt='none')

        # plot fitting line
        x = np.linspace(min(V_data) * 0.8, max(V_data) * 1.1, 5000)
        k = np.mean(P_data * V_data)
        fit_P = k / x
        ax3.plot(x, fit_P, 'r', label=r'$PV =$%2.e' % k)
        ax3.legend()
        ax3.set_title(r"Boyle's Law: $PV = k$")
        ax3.set_xlabel(r'$V$')
        ax3.set_ylabel(r'$P$')

    if gay:
        print("Plotting Gay-Lussac's Law")
        # Gay-Lussac/Amonton's Law
        # P ~ T at constant V

        T_data = []
        P_data = []  # to be used as labels
        P_err_data = []

        T_multipliers = np.linspace(0.5, 3, points)
        v_p_multipliers = np.sqrt(T_multipliers)

        for v_p_multiplier in v_p_multipliers:
            # generate new simulation with new V
            gas = base_gas.copy()
            gas['v_p'] = gas['v_p'] * np.sqrt(v_p_multiplier)  # T = T(v^2)
            _simulation_parameters = base_simulation_parameters.copy()
            _simulation_parameters['gases'] = [gas]

            returned_data = find_stats_for_simulation_parameters(
                _simulation_parameters, duration_limit=duration_limit, repeats=repeats, combine_data=True)
            P = np.mean(returned_data['pressure'])
            P_err = np.std(returned_data['pressure'])

            T = np.mean(returned_data['temperature'])  # constant
            T_data.append(T)
            P_data.append(P)
            P_err_data.append(P_err)

        P_data = np.array(P_data)
        T_data = np.array(T_data)

        ax4 = fig.add_subplot(224)
        ax4.plot(T_data, P_data, 'b.')
        ax4.set_title("Gay-Lussac's Law: $P \propto T$")
        ax4.set_xlabel(r'$T$')
        ax4.set_ylabel(r'$P$')
        ax4.errorbar(T_data, P_data, yerr=P_err_data, fmt='none')

        # plot fitting line
        x = np.linspace(min(T_data) * 0.8, max(T_data) * 1.1, 5000)
        k = np.mean(P_data / T_data)
        fit_P = k * x
        ax4.plot(x, fit_P, 'r', label=r'$P =$%2.e$T$' % k)
        fit_P = np.mean(P_data / T_data)
        ax4.legend()

        ax4.set_xlim(xmin=0)
        ax4.set_ylim(ymin=0)

    plt.tight_layout()
    plt.show()

    # generate same velocities


def find_stats_for_simulation_parameters(simulation_parameters, repeats=5, save_file='sim_stats.csv', save_data=True, duration_limit=3, combine_data=True):
    """Repeats measurements of temperature and pressure for a given set of simulation_parameters. Accepts parameters: number of repeats, file name to save data as (in csv), duration_limit for runs, and combine_data boolean. If combine_data, the data will be combined into a single dict (dict is returned instead of list, with values being the grouped values found in the original list of dicts)."""

    stats_keys = ['frame_no', 'simulated_duration', 'average_velocity', 'total_kinetic_energy',
                  'temperature', 'pressure', 'average_collisions', 'collisions on container']

    all_stats_dicts = []

    for i in range(repeats):
        _s = Simulation(**simulation_parameters)
        stats_dict = _s.run(
            _return=True, show_stats_midway=False, duration_limit=duration_limit)
        all_stats_dicts.append(stats_dict)

    keys_to_write = stats_keys + list(simulation_parameters.keys())
    # combine stats_dicts and sim_params
    data_to_write = [{**x, **simulation_parameters} for x in all_stats_dicts]

    if save_data:
        if os.path.isfile(save_file):
            # append new line
            with open(save_file, 'a') as f:
                f.write('\n')
            with open(save_file, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=keys_to_write, restval='')
                writer.writeheader()
                writer.writerows(data_to_write)
        else:
            with open(save_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=keys_to_write, restval='')
                writer.writeheader()
                writer.writerows(data_to_write)

    if combine_data:
        # combine keys in stats_dict
        returned_data = {}
        for _key in stats_keys:
            list_of_data = []
            for stats_dict in all_stats_dicts:
                if stats_dict.get(_key, None) is not None:
                    list_of_data.append(stats_dict[_key])
            returned_data[_key] = np.array(list_of_data)
    else:
        returned_data = all_stats_dicts
    return returned_data


def plot_b_against_V_g(duration_limit=3, repeats=5, velocity_seed=1):
    """Plot b against Vg for variety of Vg (Vg is volume of gas molecule). Slope *should* be Avogadro's number. Off by an order of magnitude. At least it's positive though."""
    # b = V/n - RT/p
    # constants
    N_A = 6.02214e23
    R = 8.3144598

    all_Vg = np.linspace(50, 500, 6)
    all_rg = np.cbrt(3 / 4 / np.pi * all_Vg)

    Vg_data = []
    b_data = []
    b_errors = []

    for rg, Vg in zip(all_rg, all_Vg):
        container_radius = 30
        N = 50  # number of molecules

        gases = [{'count': N, 'radius': rg, 'v_p': 20, 'mass': 1}]
        _simulation_parameters = {
            'gases': gases,
            'dimensions': 3,
            'simulation_method': 1,
            'animation': False,
            'container_radius': container_radius,
            'velocity_seed': velocity_seed
        }
        returned_data = find_stats_for_simulation_parameters(
            _simulation_parameters, duration_limit=duration_limit, repeats=repeats, combine_data=True)
        pprint.pprint(returned_data)

        # T = np.mean(returned_data['temperature'])
        # P = np.mean(returned_data['pressure'])

        for n in range(repeats):
            T = returned_data['temperature'][n]
            P = returned_data['pressure'][n]

            V = 4 / 3 * np.pi * container_radius**3
            n = N / N_A
            b = V / n - R * T / P
            b_data.append(b)
            Vg_data.append(Vg)

            # errors
            # T_err = np.std(returned_data['temperature'])
            # P_err = np.std(returned_data['pressure'])
            # b_err = np.sqrt(((T_err / T)**2 + (P_err / P))**2) * b
            # b_errors.append(b_err)
    b_data = np.array(b_data)
    Vg_data = np.array(Vg_data)

    plt.figure(3)
    print('x, y:')
    # pprint.pprint([(Vg, b) for Vg, b in zip(Vg_data, b_data)])
    print(np.array([Vg_data, b_data]).T)
    calculated_N_A = b_data / Vg_data
    plt.plot(Vg_data, b_data, 'b.')
    # plt.errorbar(all_Vg, b_data, yerr=b_errors, fmt='none')

    print('N_As:', calculated_N_A)
    # labels = ['%.1e' % _x for _x in calculated_N_A]
    # for label, x, y in zip(labels, Vg_data, b_data):
    #     plt.annotate(label, xy=(x, y), xytext=(-5, 5),
    #                  textcoords='offset points', ha='right', va='bottom')
    calc_N_A = np.mean(calculated_N_A[-3:])

    # plot best fit line
    x = np.linspace(0, 550)
    y = x * calc_N_A
    plt.plot(x, y, 'r-', label=r'$b = $%.2e $V_G$' % calc_N_A)
    plt.legend()
    plt.title(r'$b$ for varying $V_G$')
    plt.xlabel(r'$V_G$ [m$^3$]')
    plt.ylabel(r'$b$ [m$^3$ mol$^{-1}$]')

    plt.tight_layout()
    plt.show()


def find_b_graphically(duration_limit=5, repeats=5, velocity_seed=1):
    """Finds b for a varying container volumes, for N2 gas values"""
    # Plot b for a variety of container volumes (V)
    # b = V/n - RT/p
    # constants
    N_A = 6.02214e23
    R = 8.3144598

    # all_V = np.linspace(1e-9, 10e-9, 5)
    # all_r = np.cbrt(3 / 4 / np.pi * all_V)

    all_r = np.linspace(1e-9, 10e-9, 10)
    all_V = 4 / 3 * np.pi * all_r**3
    V_data = []
    b_data = []
    b_errors = []

    for r, V in zip(all_r, all_V):
        N = 50  # number of molecules

        gases = [{'count': N, 'radius': 1.55e-10,
                  'v_p': 4.09e2, 'mass': 4.50e-26}]
        _simulation_parameters = {
            'gases': gases,
            'dimensions': 3,
            'simulation_method': 2,
            'animation': False,
            'container_radius': r,
            'velocity_seed': velocity_seed
        }
        returned_data = find_stats_for_simulation_parameters(
            _simulation_parameters, duration_limit=duration_limit, repeats=repeats, combine_data=True)
        pprint.pprint(returned_data)

        # is actually constant due to seed(mean not needed)
        T = np.mean(returned_data['temperature'])
        P = np.mean(returned_data['pressure'])

        V = V
        n = N / N_A
        b = V / n - R * T / P
        b_data.append(b)
        V_data.append(V)

        T_err = np.std(returned_data['temperature'])
        P_err = np.std(returned_data['pressure'])
        b_err = np.sqrt(((T_err / T)**2 + (P_err / P))**2) * b
        b_errors.append(b_err)

    b_data = np.array(b_data)
    V_data = np.array(V_data)

    plt.figure(3)
    print('x, y:')
    print(np.array([V_data, b_data]).T)
    plt.plot(V_data, b_data, 'b.')
    plt.errorbar(V_data, b_data, yerr=b_errors, fmt='none')

    labels = ['%.1e' % _x for _x in b_data]
    for label, x, y in zip(labels, V_data, b_data):
        plt.annotate(label, xy=(x, y), xytext=(0, 5),
                     textcoords='offset points', ha='center', va='bottom')

    plt.title(r'Measured $b$ for varying container volume' )
    plt.xlabel(r'$V$ [m$^3$]')
    plt.ylabel(r'$b$ [m$^3$ mol$^{-1}$]')

    plt.tight_layout()
    plt.show()

