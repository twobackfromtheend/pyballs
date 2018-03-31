import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
import numpy as np

FRAME_INTERVAL = 16
SPEED_UP = 1  # simulates 1/FRAME_INTERVAL * SPEED_UP per frame. DOES NOT AFFECT STATS CALCULATION


class BallAnimation:
    """Contains all methods required to start and run an animation for a simulation"""
    colours = {}  # stores colours as radius:colour

    def __init__(self, simulation, speed_up=1., frame_interval=16, **kwargs):
        """Sets up animation to be run for simulation passed."""
        self.simulation = simulation
        self.speed_up = speed_up
        self.frame_interval = frame_interval
        self.balls = simulation.balls
        self.figure = plt.figure(1)
        if self.simulation.dimensions == 2:
            self.ax = self.figure.add_subplot(111, aspect='equal')
            axis_limits = self.simulation.container_radius * 1.1
            self.ax.set_xlim((-axis_limits, axis_limits))
            self.ax.set_ylim((-axis_limits, axis_limits))
            self.patches = [self.ax.text(
                0.01, 0.01, 'frame no: 0', ha='left', va='bottom', transform=self.ax.transAxes)]
        elif self.simulation.dimensions == 3:
            self.ax = self.figure.add_subplot(
                111, projection='3d', aspect='equal')

        self.frame_no = 0

    def init_figures(self):
        """Initialises figures for 2D plot"""
        self.container = plt.Circle(
            (0, 0), self.simulation.container_radius, ec='b', fill=False, ls='solid')
        self.ax.add_artist(self.container)

        for b in self.balls[:-1]:
            self.ax.add_patch(b.patch)
            colour = BallAnimation.get_colour(b)
            b.patch.set_facecolor(colour)
            self.patches.append(b.patch)
        return self.patches

    def next_frame(self, frame_no):
        """Runs simulation for frame_duration, replot frame"""
        try:
            frame_duration = self.speed_up / self.frame_interval
            # do all collisions in frame_duration
            self.simulation.simulation_method(frame_duration)

            if frame_no % 15 == 10:
                self.simulation.collect_velocities()

            if frame_no % 100 == 50:
                self.simulation.show_stats(frame_no, plot=True)

            if self.simulation.dimensions == 2:
                self.ax.add_artist(self.container)
                # update plot texts
                self.patches[0].set_text('frame no: %s' % frame_no)
                self.simulation.time_simulated += frame_duration
                return self.patches
            else:
                # plot all balls in 3D
                self.ax.clear()
                for ball in self.simulation.balls:
                    self.plot_sphere(ball)
                self.ax.set_aspect('equal')
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])
                self.simulation.time_simulated += frame_duration

        except:
            # try except to work around Spyder matplotlib error catching
            import traceback
            traceback.print_exc()
            raise Exception('Exception encountered in animation.')

    def run(self):
        """Runs simulation animation (FuncAnimation)."""
        if self.simulation.dimensions == 2:
            fig = plt.figure(1)

            self.anim = animation.FuncAnimation(
                fig,
                self.next_frame,
                init_func=self.init_figures,
                interval=16,
                blit=True)
            plt.show()

        elif self.simulation.dimensions == 3:
            fig = plt.figure(1)

            self.anim = animation.FuncAnimation(
                fig,
                self.next_frame,
                interval=16)
            plt.show()
        elif self.simulation.dimensions > 3:
            raise Exception(
                'Plotting more dimensions than spatially available in universe')
        else:
            raise Exception('Unsupported dimensions for animation.')

    def plot_sphere(self, ball):
        """Plots sphere surface for given ball"""
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        x = ball.radius * np.outer(np.cos(u), np.sin(v)) + ball.position[0]
        y = ball.radius * np.outer(np.sin(u), np.sin(v)) + ball.position[1]
        z = ball.radius * np.outer(np.ones(np.size(u)),
                                   np.cos(v)) + ball.position[2]

        alpha = 0.1 if ball.is_container else 0.5

        colour = BallAnimation.get_colour(ball)
        self.ax.plot_surface(x, y, z, rstride=8, cstride=8,
                             alpha=alpha, linewidth=0, color=colour)

    @staticmethod
    def get_colour(ball):
        """Generates random colour for radius, adds colour to BallAnimation.colours dict so balls of the same radius are coloured the same."""
        # get colour based on radius
        try:
            return ball.colour
        except AttributeError:
            if ball.radius in BallAnimation.colours:
                ball.colour = BallAnimation.colours[ball.radius]
            else:
                # generate new colour for radius
                BallAnimation.colours[ball.radius] = np.random.random(3)
                ball.colour = BallAnimation.colours[ball.radius]
            return ball.colour


class StatsAnimation:

    def __init__(self, simulation):
        pass
