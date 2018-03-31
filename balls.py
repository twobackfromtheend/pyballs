import numpy as np
import matplotlib.pyplot as plt
from anim import BallAnimation
from scipy.stats import maxwell
import numpy.random as npr
from scipy.stats import maxwell, rayleigh
import time

import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

COLLECT_STATS = True
k_b = 1.38064852e-23


class Ball:
    """
    Ball objects are used to simulate gas atoms. Balls are created in the Gas class, passing a position, velocity, mass, radius, and gas to create each ball.
    """
    i = 0
    balls = {}

    def __init__(self, position, velocity, mass=1., radius=1., gas=None):
        """Creates ball object with given position, velocity, [mass, radius]. Creates 2D patch."""
        self.gas = gas
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius

        self.mass = mass

        self.is_container = True if mass == -1 else False
        self.next_collision = None  # [time, self, collidee]

        self.collision_count = 0
        self.patch = plt.Circle(self.position, self.radius, fc='r')
        self.i = Ball.i
        Ball.balls[self.i] = self
        Ball.i += 1

    def __repr__(self):
        return "Ball.%s" % self.i

    def move(self, time):
        """Move ball by time. Updates patch centre and next_collision."""
        # move ball by time, update patch centre
        self.position += self.velocity * time
        self.patch.center = self.position

        # update next collision when moving
        if self.next_collision is not None:
            self.next_collision[0] -= time

    def time_to_collision(self, other, dt_limit=0, use_isclose=False):
        """Calculates time to collision between the 2 balls passed as parameters. Returns None if no collision in the future. Also returns None if next collision happens within the next dt_limit to prevent recollision of overlapping balls (recollision would make separating balls head toward each other and overlap even more, leading to more recollisions). dt_limit is 0 by default, but can be set to 1e-10 for normal animatable simulations."""

        dv = self.velocity - other.velocity
        dr = self.position - other.position
        dv2 = dv.dot(dv)
        if dv2 == 0:
            # parallel velocities
            return None
        dr2 = dr.dot(dr)
        if other.is_container or self.is_container:
            dR2 = (self.radius - other.radius)**2
        else:
            dR2 = (self.radius + other.radius)**2
        _b = dv.dot(dr)  # not actually b, (simplified).
        determinant = _b**2 - dv2 * (dr2 - dR2)

        if determinant < 0:
            # no collisions, past or future
            return None
        else:
            d_sqrt = np.sqrt(determinant, dtype=np.float64)
            all_dt = np.zeros(2, dtype=np.float64)
            all_dt[0] = (-_b + d_sqrt) / dv2
            all_dt[1] = (-_b - d_sqrt) / dv2
            # find only positive dt's, return minimum.
            # print(dt1, dt2)
            dt_limit = dt_limit
            if (all_dt < 0).all() and (other.is_container or self.is_container):
                # print('Ball outside container')
                # print(all_dt)
                # print(self.i, other.i)
                # print(self.position, self.velocity)
                # print(other.position, other.velocity)
                raise RuntimeError('Ball outside container')
            all_dt = all_dt[all_dt >= dt_limit]
            if all_dt.size == 0:
                return None
            dt = np.amin(all_dt)

            # these comments prints dt only if nothing will be returned.
            # if dt is None and (0 < dt1 < dt_limit or 0 < dt2 < dt_limit):
            #     print(
            #         'Small dt ignored: %5.3e or %5.3e, current limit: %1.0e' % (dt1, dt2, dt_limit))
            # these comments print both dts
            # if 0 < dt1 < dt_limit or 0 < dt2 < dt_limit:
            #     print('(Greedy) Small dt ignored: %5.3e or %5.3e, current limit: %1.0e' % (
            #         dt1, dt2, dt_limit))
            #     print(self.i)
            if use_isclose and dt is not None and np.allclose(dt, 0.):
                # print('np.allclose: %5.3e or %5.3e, current limit: %1.0e' % (dt1, dt2, dt_limit))
                return None
            return dt

    @staticmethod
    def find_first_collisions(simulation):
        """Finds first collisions for all balls in simulation. Enables do_simulation_for_t to not have to check for next_collision time existing every frame."""
        ball_count = len(simulation.balls)
        for _i in range(ball_count):
            collider = simulation.balls[_i]
            for _j in range(_i + 1, ball_count):
                collidee = simulation.balls[_j]

                _time_to_collision = collider.time_to_collision(collidee)
                if _time_to_collision is not None:
                    # update times to collisions
                    if collider.next_collision is None or _time_to_collision < collider.next_collision[0]:
                        collider.next_collision = [
                            _time_to_collision, collider, collidee]
                    if collidee.next_collision is None or _time_to_collision < collidee.next_collision[0]:
                        collidee.next_collision = [
                            _time_to_collision, collidee, collider]

    def check_if_accepted_collision(self, other, t=None):
        """Checks if collision between self and other (at optional time t later) is valid. Returns True if collision should be done, False otherwise."""
        if t is None:
            # check for overlapping recolliding balls
            if self.is_container or other.is_container:
                ball = self if other.is_container else other
                if np.dot(ball.velocity, ball.position) < 0:
                    # already moving back to centre
                    # print('1', self.i, other.i, ball.velocity, ball.position)
                    return False
            else:
                relative_positions = self.position - other.position
                relative_velocities = self.velocity - other.velocity
                if np.dot(relative_positions, relative_velocities) > 0:
                    # already moving apart
                    # print('2', self.i, other.i)
                    return False
            return True
        else:
            # theoretically move stuff (do not implement actual move)
            # then do check
            new_self_position = self.position + self.velocity * t
            new_other_position = other.position + other.velocity * t

            if other.is_container:
                # self is ball
                if np.dot(self.velocity, new_self_position) < 0:
                    return False
            elif self.is_container:
                # other is ball
                if np.dot(other.velocity, new_other_position) < 0:
                    return False
            else:
                relative_positions = new_self_position - new_other_position
                relative_velocities = self.velocity - other.velocity
                if np.dot(relative_positions, relative_velocities) > 0:
                    return False
            return True

    def collide(self, other):
        """Updates velocities of colliding balls."""
        self.collision_count += 1
        other.collision_count += 1

        # calculate resulting velocities from ball collision
        if self.is_container or other.is_container:
            # ball on cont action
            separation_vector = self.position - other.position
            normalised_separation_vector = separation_vector / \
                (np.sqrt(separation_vector.dot(separation_vector)))

            # only touch the ball
            ball = self if not self.is_container else other
            # cont = self if self.is_container else other

            ball_parallel = ball.velocity.dot(
                normalised_separation_vector) * normalised_separation_vector
            ball_perpendicular = ball.velocity - ball_parallel

            new_ball_velocity = ball_perpendicular - ball_parallel
            old_ball_velocity = ball.velocity  # used later for stats collection

            # update ball velocity
            ball.velocity = new_ball_velocity

            # stats collection
            if COLLECT_STATS:
                velocity_change = old_ball_velocity - new_ball_velocity
                momentum_change = ball.mass * \
                    np.sqrt(velocity_change.dot(velocity_change))
                return momentum_change

        else:
            # ball on ball action
            separation_vector = self.position - other.position
            normalised_separation_vector = separation_vector / \
                (np.sqrt(separation_vector.dot(separation_vector)))

            m1 = self.mass
            m2 = other.mass

            # split velocities into surface normal parallel and perpendicular
            # components
            v1_parallel = self.velocity.dot(
                normalised_separation_vector) * normalised_separation_vector
            v1_perpendicular = self.velocity - v1_parallel

            v2_parallel = other.velocity.dot(
                normalised_separation_vector) * normalised_separation_vector
            v2_perpendicular = other.velocity - v2_parallel

            # do 1D dimension on parallel components
            v1_parallel_new = ((m1 - m2) * v1_parallel + 2 *
                               m2 * v2_parallel) / (m1 + m2)
            v2_parallel_new = ((m2 - m1) * v2_parallel + 2 *
                               m1 * v1_parallel) / (m1 + m2)

            # update velocities
            self.velocity = v1_perpendicular + v1_parallel_new
            other.velocity = v2_perpendicular + v2_parallel_new


class Gas:
    """
    Gas class used to generate and simulate a group of balls with the same characteristics.
    """

    def __init__(self, simulation, radius=0.1, mass=None, v_p=1, count=5):
        self.simulation = simulation

        if self.simulation.dimensions == 2:
            self.mass = mass if mass is not None else radius**2
        if self.simulation.dimensions == 3:
            # set mass using N2 example and assuming constant density. N2 has radius of 155pm (1.55e-10 m) and mass of 4.65e-23g (4.65e-26kg)
            # density = 4.65e-26 / (4 / 3 * np.pi * (1.55e-10)**3)
            # self.mass = 4 / 3 * np.pi * radius**3 * density
            # 4.65e-26 * (radius / 1.55e-10)**3
            self.mass = mass if mass is not None else radius**3

        self.radius = radius
        self.count = count
        self.v_p = v_p
        self.balls = []

        # generate position, velocity variables
        positions = self.generate_positions()
        velocities = self.generate_velocities()
        # create balls
        for i in range(count):
            ball = Ball(
                position=positions[i],
                velocity=velocities[i],
                mass=self.mass,
                radius=self.radius,
                gas=self
            )
            self.balls.append(ball)

    def __repr__(self):
        return 'Gas([r=%.1e, m=%.1e, v_p=%.1e, count=%s])' % (self.radius, self.mass, self.v_p, self.count)

    def generate_positions(self):
        """Randomly generate positions of balls until required number is reached, with none overlapping or outside the container."""
        # randomly generate necessary positions
        positions = (npr.rand(self.count, self.simulation.dimensions) -
                     0.5) * 2 * (self.simulation.container_radius - self.radius)
        MAX_ITER = 10000

        _iteration = 0
        # get non-overlapping balls
        while True:
            # check for exceeding container
            distances_from_centre = np.sqrt(np.sum(positions**2, axis=1))
            positions[distances_from_centre >
                      self.simulation.container_radius - self.radius] = np.nan

            # check for overlaps with previous gases
            for gas in self.simulation.gases:
                for ball in gas.balls:
                    for _i in range(self.count):
                        # detects overlaps between ball and positions[_i]
                        separation = ball.position - positions[_i]
                        if separation.dot(separation) < (ball.radius + self.radius)**2:
                            # if overlapping, set positions to np.nan
                            positions[_i] = np.nan

            # check for overlaps with self
            for _i in range(self.count):
                if not np.isnan(positions[_i, 0]):
                    for _j in range(_i + 1, self.count):
                        if not np.isnan(positions[_j, 0]):
                            separation = positions[_i] - positions[_j]
                            if separation.dot(separation) < (2 * self.radius)**2:
                                # if overlapping set positions to np.nan
                                positions[_j] = np.nan

            nans_mask = np.isnan(positions)
            count_left = np.count_nonzero(nans_mask)
            if count_left == 0:
                # no new positions needed
                break
            elif _iteration > MAX_ITER:
                raise Exception(
                    'Could not generate %s balls of radii %s into a container of radius %s in %s iterations. %s ungenerated.' % (self.count, self.radius, self.simulation.container_radius, MAX_ITER, int(count_left / self.simulation.dimensions)))

            # generate new positions for nans
            positions[nans_mask] = (npr.rand(count_left) -
                                    0.5) * 2 * (self.simulation.container_radius - self.radius)
            _iteration += 1

        return positions

    def generate_velocities(self):
        """Generate velocities with mean velocity using a Rayleigh distribution for 2D and a Maxwell distribution for 3D."""
        # randomly generate velocities with a Maxwell-Boltzmann dist.

        if self.simulation.velocity_seed is not None:
            # used to keep temperature constant
            npr.seed(self.simulation.velocity_seed)
        # generate velocities in polar coordinates
        # (so maxwell dist can be used)
        if self.simulation.dimensions == 2:
            # rayleigh has mean of np.sqrt(pi/2)
            # for a rayleigh dist, mean = a * sqrt(pi/2), mode = a.
            # a is passed as the scale factor.
            radial_speeds = rayleigh.rvs(scale=self.v_p, size=self.count)
            phi = npr.random(size=self.count) * 2 * np.pi

            velocities = (
                np.array([np.cos(phi), np.sin(phi)]) * radial_speeds).T

        elif self.simulation.dimensions == 3:
            # for a maxwell dist, mean = 2a * sqrt(2/pi), mode = sqrt(2) * a.
            # a is passed as the scale factor.
            radial_speeds = maxwell.rvs(
                scale=self.v_p / np.sqrt(2), size=self.count)
            theta = npr.random(size=self.count) * np.pi
            phi = npr.random(size=self.count) * 2 * np.pi

            velocities = (
                np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]) * radial_speeds).T
        # rerandomise seed
        npr.seed()
        return velocities


class Simulation:
    """Creates a simulation with the given gas parameters and links all parts required for a simulation (animation, gases, balls)."""

    def __init__(self, gases=[{}], container_radius=1, dimensions=2, animation=True, simulation_method=2, sim_method_3_checks_per_frame=50, m3=None, duration_limit=None, frame_limit=None, velocity_seed=None, **kwargs):
        """Creates Simulation object with passed parameters. Accepts further kwargs to pass to animation creation."""
        # randomise numpy seed so positions will not be affected by seed.
        npr.seed()
        # duration and frame limits defaults are set in run()
        self.duration_limit = float(
            duration_limit) if duration_limit is not None else duration_limit
        self.frame_limit = int(
            frame_limit) if frame_limit is not None else frame_limit

        if m3 is not None:
            self.sim_method_3_checks_per_frame = m3
        else:
            self.sim_method_3_checks_per_frame = sim_method_3_checks_per_frame

        self.velocity_seed = velocity_seed

        self.container_radius = container_radius
        self.dimensions = dimensions
        self.container = Ball(position=[0 for _i in range(dimensions)], velocity=[
                              0 for _i in range(dimensions)], mass=-1, radius=container_radius)
        self.gases = []
        self.balls = []

        for gas_parameters in gases:
            gas = Gas(simulation=self, **gas_parameters)
            self.gases.append(gas)
        x = [ball for gas in self.gases for ball in gas.balls] + [self.container]

        self.balls += x

        # velocities is mass:velocities
        self.stats = {'pressures': [], 'velocities': {}}
        self.time_simulated = 0

        print('Generated %s balls:' % len(self.balls))
        Ball.find_first_collisions(self)
        # for ball in self.balls:
        #     print(ball.next_collision)

        self.simulation_method = simulation_method
        if simulation_method == 1:
            self.simulation_method = self.do_simulation_for_t
        elif simulation_method == 2:
            self.simulation_method = self.do_simulation_for_t_2
        elif simulation_method == 3:
            self.simulation_method = self.do_simulation_for_t_3

        # passes on animation variables
        self.anim = BallAnimation(self, **kwargs) if animation else None

    def find_next_collision(self):
        """Finds the next collision. Returns the time to collision, and the two balls involved in said collision."""
        collisions = []  # (time, collider, collidee)
        for _i in range(len(self.balls) - 1):
            collider = self.balls[_i]
            for _j in range(_i + 1, len(self.balls)):
                collidee = self.balls[_j]
                try:
                    collision_time = collider.time_to_collision(collidee)
                except RuntimeError:
                    # catch ball outside container.
                    # never happens here
                    collision_time = None

                if collision_time is not None:
                    collisions.append((collision_time, collider, collidee))

        # sort collisions, in reverse order to optimise popping of the last one
        collisions.sort(key=lambda x: x[0], reverse=True)
        next_collision_time, collider, collidee = collisions[-1]
        # check if it's a wanted collision
        accepted_collision = False
        while next_collision_time < 1e-10 and not accepted_collision:
            # collisions after 1e-10 don't need to be checked for some reason.
            if collider.check_if_accepted_collision(collidee, next_collision_time):
                accepted_collision = True
            else:
                collisions.pop()
            next_collision_time, collider, collidee = collisions[-1]
        return next_collision_time, collider, collidee

    def do_simulation_for_t(self, t):
        """Runs simulation for time t. This method calculates next collision for all balls each time a collision happens."""
        next_collision_time, collider, collidee = self.find_next_collision()
        _frame_duration_left = t
        while next_collision_time <= _frame_duration_left:
            for ball in self.balls:
                ball.move(next_collision_time)
            _frame_duration_left -= next_collision_time
            momentum_change = collider.collide(collidee)
            if momentum_change:
                self.stats['pressures'].append(momentum_change)

            next_collision_time, collider, collidee = self.find_next_collision()

        for ball in self.balls:
            ball.move(_frame_duration_left)

    def update_ball_next_collision(self, ball, ttl=0):
        """Update next_collision attribute for given ball. ttl parameter should be set to 1 for updates after actual collisions, as it allows updating of the next_collision attribute for other balls in the simulation if the initial, just collided, ball is involved."""
        collider = ball
        # print('hi', ball.i)
        for collidee in self.balls:
            if collider is collidee:
                continue
            if collidee.next_collision and ttl > 0 and ball in collidee.next_collision:
                # update collidee's next_collision as this collision may not
                # happen after the collision
                collidee.next_collision = None
                self.update_ball_next_collision(collidee)

            try:
                time_to_collision = collider.time_to_collision(collidee)
            except RuntimeError:
                # force collision
                print('FORCED COLLISION FOR ROGUE BALL')
                if collider.check_if_accepted_collision(collidee):
                    collider.collide(collidee)
                time_to_collision = None

            if time_to_collision is None:
                continue

            if collider.check_if_accepted_collision(collidee, time_to_collision):
                # check for lower time_to_collision (or at least not None),
                # and not container (do not set container.next_collision)
                if collider.next_collision is None or time_to_collision < collider.next_collision[0]:
                    collider.next_collision = [
                        time_to_collision, collider, collidee]

                if collidee.next_collision is None or time_to_collision < collidee.next_collision[0]:
                    collidee.next_collision = [
                        time_to_collision, collidee, collider]

    def do_simulation_for_t_2(self, t):
        """Runs simulation for time t. This method updates next_collision only for balls involved in the collision (and balls with these collided balls in their own next_collision)."""
        _frame_duration_left = t
        next_collisions = [_ball.next_collision for _ball in self.balls[
            :-1] if _ball.next_collision is not None]
        next_collisions.sort(key=lambda x: x[0], reverse=True)

        next_collision_time, collider, collidee = next_collisions[-1]

        while next_collision_time <= _frame_duration_left:
            for ball in self.balls[:-1]:
                ball.move(next_collision_time)
            _frame_duration_left -= next_collision_time
            momentum_change = collider.collide(collidee)
            if momentum_change:
                self.stats['pressures'].append(momentum_change)

            self.update_ball_next_collision(collider, ttl=1)
            self.update_ball_next_collision(collidee, ttl=1)
            for ball in self.balls[:-1]:
                if ball.next_collision is None:
                    self.update_ball_next_collision(ball)
                    # print('Updating ball for none next_collision', ball.i, ball.next_collision)
            accepted_collision = False
            next_collisions = [_ball.next_collision for _ball in self.balls[
                :-1] if _ball.next_collision is not None]
            next_collisions.sort(key=lambda x: x[0], reverse=True)
            next_collision_time, collider, collidee = next_collisions[-1]

            while next_collision_time < 1e-10 and not accepted_collision:
                # collisions after 1e-10 don't need to be checked for some
                # reason.
                if collider.check_if_accepted_collision(collidee, next_collision_time):
                    accepted_collision = True
                else:
                    next_collisions.pop()

                    collider.next_collision = None
                    self.update_ball_next_collision(collider, ttl=0)
                    self.update_ball_next_collision(collidee, ttl=0)
                next_collision_time, collider, collidee = next_collisions[-1]

        for ball in self.balls:
            ball.move(_frame_duration_left)

    def do_simulation_for_t_3(self, t, dt=None, do_separation_check=True):
        """Runs simulation for time t, checking if collisions have occured every dt duration. This just does a distance check between all balls every dt. By default, dt is t/50 (thus doing 50 checks each frame)"""

        if dt is None:
            dt = t / self.sim_method_3_checks_per_frame
        _frame_duration_left = t
        while _frame_duration_left > dt:
            # simulate duration dt, check for collisions, loop.
            ball_count = len(self.balls)
            for i in range(ball_count - 1):
                ball_1 = self.balls[i]

                # check collisions with other balls
                for j in range(i + 1, ball_count - 1):
                    ball_2 = self.balls[j]

                    separation = ball_1.position - ball_2.position
                    if separation.dot(separation) <= (ball_1.radius + ball_2.radius)**2:
                        # check for overlapping recolliding balls
                        if ball_1.check_if_accepted_collision(ball_2):
                            momentum_change = ball_1.collide(
                                ball_2)  # collide balls

                            if COLLECT_STATS and momentum_change:
                                self.stats['pressures'].append(momentum_change)

                # check collisions with container
                container = self.balls[-1]
                if ball_1.position.dot(ball_1.position) >= (container.radius - ball_1.radius)**2:
                    if ball_1.check_if_accepted_collision(container):
                        momentum_change = ball_1.collide(
                            container)  # collide balls

                        if COLLECT_STATS and momentum_change:
                            self.stats['pressures'].append(momentum_change)

            for ball in self.balls:
                ball.move(dt)
            _frame_duration_left -= dt

        for ball in self.balls:
            ball.move(_frame_duration_left)

    def get_velocities(self, return_type='list'):
        """Returns an NumPy array of velocities [m s-1] of all balls in the simulation (shape: (no. of balls, no. of dimensions))"""
        if return_type == 'list':
            velocities = []
            for ball in self.balls[:-1]:
                velocities.append(ball.velocity)
            return np.array(velocities)
        else:  # return dict
            velocities = {}  # mass: velocity
            for gas in self.gases:
                if gas.mass not in velocities:
                    velocities[gas.mass] = []
                for ball in gas.balls:
                    velocities[gas.mass].append(ball.velocity)
                velocities[gas.mass] = np.array(velocities[gas.mass])
            return velocities

    def get_kinetic_energies(self):
        """Returns a NumPy array of kinetic energies [J] of all balls in the simulation (shape: no. of balls)."""
        velocities = self.get_velocities()

        masses = self.get_masses()
        kinetic_energies = 1 / 2 * masses * \
            (velocities * velocities).sum(axis=1)
        return kinetic_energies

    def get_momentums(self):
        """Returns NumPy array of the momentums [kg m s-1] of all balls in the simulation."""
        velocities = self.get_velocities()

        masses = []
        for ball in self.balls[:-1]:
            masses.append(ball.mass)
        masses = np.array(masses)
        print(masses)
        print(velocities)
        momentums = (velocities.T * masses).T
        return momentums

    def get_pressure(self):
        """Returns the pressure [Pa] in the simulation. Returns None if time_simulated is 0"""
        if self.time_simulated == 0:
            return None
        if self.dimensions == 2:
            area = 2 * np.pi * self.container_radius
            pressure = np.array(self.stats['pressures']).sum(
            ) / self.time_simulated / area  # faster than Python sum(x)
        elif self.dimensions == 3:
            area = 4 * np.pi * self.container_radius**2
            pressure = np.array(self.stats['pressures']).sum(
            ) / self.time_simulated / area
        # reset pressure for testing purposes
        # self.stats['pressures'] = []
        # self.time_simulated = 0.
        return pressure

    def get_masses(self):
        """Returns the masses [kg] of all balls in the simulation."""
        masses = []
        for ball in self.balls[:-1]:
            masses.append(ball.mass)
        return np.array(masses)

    def calculate_temp_and_ke(self):
        """Returns the temperature [K], total_kinetic_energy and average_kinetic_energy of balls [J] in the simulation."""
        kinetic_energies = self.get_kinetic_energies()
        total_kinetic_energy = kinetic_energies.sum()
        average_kinetic_energy = total_kinetic_energy / \
            (len(self.balls) - 1)
        temperature = 2. / self.dimensions * average_kinetic_energy / k_b
        return temperature, total_kinetic_energy, average_kinetic_energy

    def collect_velocities(self):
        """Collects velocity data and appends it to the self.stats['velocities'] dictionary for each gas. (This dictionary containes key:value pairs of gas.mass:velocities)"""
        _velocities = self.get_velocities(return_type='dict')
        for gas in self.gases:
            if gas.mass not in self.stats['velocities']:
                self.stats['velocities'][gas.mass] = np.array([])
            # get radial speeds
            velocities = np.sqrt(
                (_velocities[gas.mass] * _velocities[gas.mass]).sum(axis=1))
            self.stats['velocities'][gas.mass] = np.concatenate(
                (self.stats['velocities'][gas.mass], velocities))

    def get_collision_counts(self):
        """Returns NumPy array of collision counts for all balls, and collision count for container"""
        collision_counts = []
        for ball in self.balls[:-1]:
            collision_counts.append(ball.collision_count)
        return np.array(collision_counts), self.balls[-1].collision_count

    def get_stats(self, frame_no=None):
        """Used to return stats for a specific simulation. Useful for when collecting data."""
        temperature, total_kinetic_energy, average_kinetic_energy = self.calculate_temp_and_ke()
        velocities = self.get_velocities()
        average_velocity = np.mean(
            np.sqrt((velocities * velocities).sum(axis=1)))
        pressure = self.get_pressure()
        collisions, container_collisions = self.get_collision_counts()
        average_collisions = collisions.mean()
        stats_dict = {'frame_no': frame_no,
                      'simulated_duration': self.time_simulated,
                      'average_velocity': average_velocity,
                      'total_kinetic_energy': total_kinetic_energy,
                      'temperature': temperature,
                      'pressure': pressure,
                      'average_collisions': average_collisions,
                      'collisions on container': container_collisions,
                      }
        return stats_dict

    def show_stats(self, frame_no, print_stats=True, plot=False):
        """Prints stats of simulation.
        Current stats implemented include: kinetic energy, temperature, pressure (measured since last stats calculation)."""
        temperature, total_kinetic_energy, average_kinetic_energy = self.calculate_temp_and_ke()
        velocities = self.get_velocities()
        average_velocity = np.mean(
            np.sqrt((velocities * velocities).sum(axis=1)))
        pressure = self.get_pressure()
        collisions, container_collisions = self.get_collision_counts()
        average_collisions = collisions.mean()
        if print_stats:
            print('\nSTATS AT FRAME %s' % frame_no)
            print('AVERAGE VEL: %s' % average_velocity)
            print('KE: %s' % total_kinetic_energy)
            print('TEMP: %s' % temperature)
            print('PRESSURE: %s' % pressure)
            print('AVERAGE COLLISIONS: %s' % average_collisions)
            print('COLLISIONS ON CONTAINER: %s' % container_collisions)
        # print('MOMENTUM COMPONENTS SUM: %s' % self.get_momentums().sum(axis=0))

        if plot:
            # equipartition
            k_bT = 1.38 * 10**-23 * temperature
            # show velocities in histogram
            plt.figure(2)
            plt.gca().clear()

            bins = 30 * len(self.gases)
            data = [self.stats['velocities'].get(
                gas.mass, []) for gas in self.gases]
            color = [np.random.random(3) for gas in self.gases]
            # print(data)
            all_data = np.concatenate(data)
            if data and all_data.any():
                max_data = all_data.max()
                min_data = all_data.min()
                bin_width = (max_data - min_data) / bins
                label = ['Norm. count for m = %.2E' %
                         gas.mass for gas in self.gases]
                if len(self.gases) > 1:
                    n, bin_edges, _ = plt.hist(
                        data, bins=bins, histtype='bar', stacked=True, normed=True, color=color, label=label)
                    n = n[-1]  # take height of last gas
                    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2

                    normed_coeff = 1 / (bin_width * len(all_data))
                    # normally len(all_data) = sum(n), but now bin_width * sum(n) = 1, and sum(n) = len(all_data) * c.  c = 1 / (bin_width * len(all_data))
                    # print('norm', normed_coeff)
                    # print((n / normed_coeff).sum(), len(all_data))
                    # this checks that the unnormalised n's sum equals the number of data.
                    yerr = np.sqrt(n / normed_coeff) * normed_coeff
                    plt.errorbar(bin_centres, n, yerr=yerr, fmt='none')
                else:
                    data = data[0]
                    n, bin_edges, _ = plt.hist(
                        data, bins=bins, normed=True, label=label)
                    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
                    normed_coeff = 1 / (bin_width * len(all_data))
                    yerr = np.sqrt(n / normed_coeff) * normed_coeff
                    plt.errorbar(bin_centres, n, yerr=yerr, fmt='none')

                x = np.linspace(0, max_data, 10000)
                # integral of y = x e^(-1/2 x^2) from 0 to infinity = 1
                # to undo the transformation (squeeze along x axis) of the
                # m/kbT term, stretch y-axis by m/kbT
                total_y = None
                for gas in self.gases:
                    m = gas.mass
                    # how many balls are of this gas out of all the balls
                    normalisation = len(gas.balls) / (len(self.balls) - 1)

                    if self.dimensions == 2:
                        y = normalisation * x * \
                            np.exp(- x**2 / 2 * m / (k_bT)) * (m / (k_bT))
                        # normalisation involved multiplying y by A. See:
                        # http://www.wolframalpha.com/input/?i=integrate+y+%3D+++sqrt(2+%2Fpi)+*+x%5E2+*+e%5E(-+x%5E2+%2F+2+*A)+*+(A%5E(3%2F2))+from+x%3D0+to+infinity
                    elif self.dimensions == 3:
                        y = normalisation * \
                            np.sqrt(2 / np.pi) * x**2 * np.exp(- x**2 /
                                                               2 * m / (k_bT)) * (m / (k_bT))**(3 / 2)
                        # N.B. normalisation for A = m/k_bT for 3 dimensions is by multiplying y by A^(3/2)
                        # http://www.wolframalpha.com/input/?i=integrate+y+%3D++++x+*e%5E(-+x%5E2+%2F+2+*+A)+*(A))+from+x%3D0+to+infinity

                    plt.plot(x, y, 'b', label='MB dist for m = %.2E' %
                             gas.mass)
                    if total_y is None:
                        total_y = y
                    else:
                        total_y += y
                if len(self.gases) > 1:
                    plt.plot(x, total_y, 'r', label='Sum of MBs')
                plt.legend()
                plt.title('Speed distribution')
                plt.xlabel(r'Speed of particle [m s$^{-1}$]')
                plt.ylabel('Normalised count')

                print('Velocity data collected:', len(all_data))
                plt.tight_layout()
                plt.show()

    def run(self, duration_limit=30, frame_limit=100060, frame_duration=None, _return=None, show_stats_midway=5000, plot_velocity_dist=True):
        """Runs simulation till limit. Runs animation if simulation.anim exists. Pass a duration_limit in seconds to stop the simulation after that amount of time has elapsed (or a frame_limit). Returns temperature and pressure of simulation if _return is True. show_stats_midway can be passed as a number to show stats every x frames. It cannot be less than 50."""
        if self.anim:
            self.anim.run()
        else:
            duration_limit = self.duration_limit if self.duration_limit is not None else duration_limit
            frame_limit = self.frame_limit if self.frame_limit is not None else frame_limit

            if frame_duration is None:
                # calculate frame duration as function of velocity and container radius
                # v_p is already a speed (scalar)
                fastest_v_p = max(gas.v_p for gas in self.gases)
                frame_duration = self.container_radius / fastest_v_p / 50
                print('Set frame duration as %1.2es' % frame_duration)
            start_time = time.time()
            frame_no = 0
            try:
                while True:
                    if (duration_limit is not None and (time.time() - start_time) > duration_limit) or (frame_limit is not None and frame_no > frame_limit):
                        if not _return:
                            print('Gases:', [gas for gas in self.gases])
                            self.show_stats(frame_no)
                            print(
                                'Limit reached at frame %s. Simulation ending.' % frame_no)
                            print('Simulated %s particles for a simulated duration of %1.2es.' % (
                                len(self.balls) - 1, self.time_simulated))
                            break
                        else:
                            self.show_stats(frame_no)
                            print('Simulated %s particles for a simulated duration of %1.2es.' % (
                                len(self.balls) - 1, self.time_simulated))
                            return self.get_stats(frame_no=frame_no)

                    self.simulation_method(frame_duration)
                    self.time_simulated += frame_duration
                    frame_no += 1

                    if frame_no % 15 == 10:
                        self.collect_velocities()
                    if show_stats_midway:
                        if frame_no % show_stats_midway == 50:
                            self.show_stats(frame_no, plot=plot_velocity_dist)
            except:
                import traceback
                traceback.print_exc()
                self.show_stats(frame_no)
                print('Error at frame %s. Simulation ending.' % frame_no)
                print('Simulated %s particles for a simulated duration of %1.2es.' % (
                    len(self.balls) - 1, self.time_simulated))
                raise Exception('Exception encountered in running simulation.')


class ArraySimulation(Simulation):

    def __init__(self, simulation):
        self.container_radius = simulation.container_radius
        self.dimensions = simulation.dimensions
        self.container = {'position': [0 for _i in range(self.dimensions)], 'velocity': [
            0 for _i in range(self.dimensions)], 'mass': -1, 'radius': self.container_radius}

        self.balls = []
        x = []
        for ball in simulation.balls:
            ball_dict = {('pos' + str(i))                         : ball.position[i] for i in range(self.dimensions)}
            ball_dict.update(
                {('vel' + str(i)): ball.velocity[i] for i in range(self.dimensions)})
            ball_dict['radius'] = ball.radius
            ball_dict['mass'] = ball.mass
            ball_dict['is_container'] = ball.is_container
            ball_dict['patch'] = ball.patch
            ball_dict['i'] = ball.i

            self.balls.append(ball_dict)
        import pandas as pd
        self.balls = pd.DataFrame.from_records(self.balls, index='i')
        print(self.balls)
        return
        x = [ball for gas in self.gases for ball in gas.balls] + [self.container]

        self.balls += x

        # self.stats = {'pressures': [], 'velocities': []}
        # self.time_simulated = 0

        print('balls:', len(self.balls))
        Ball.find_first_collisions(self)
        self.anim = BallAnimation(self)

        del simulation


if __name__ == "__main__":
    pass
    # balls = [Ball([-0.3, 0.0], [0, 0]), Ball([0.3, 0], [-1, 0])]
    # anim = Animation(balls)
    #    s = Simulation(gases=[{'count': 5, 'radius': 0.1}], simulation_method=3)
    #    s = Simulation(gases=[{'count': 5, 'radius': 0.1}])
    #    s = Simulation(gases=[{'count': 5, 'radius': 0.1}], animation=None)
    #    s = Simulation(gases=[{'count': 3, 'radius': 0.1}, {'count': 2, 'radius': 0.3}])
    # s = Simulation(gases=[{'count': 50, 'radius': 0.04}], simulation_method=2)

    # s = Simulation(gases=[{'count': 50, 'radius': 0.004}, {'count': 4, 'radius': 0.04, 'v_p': 0.01}])

    # s = Simulation(gases=[{'count': (i + 1) * 5, 'radius': j / 200}
    # for i, j in enumerate(range(8, 2, -1))], simulation_method=3)

    #    s = Simulation(gases=[{'count': 3, 'radius': 0.1},{'count': 3, 'radius': 0.05},{'count': 2, 'radius': 0.3}], dimensions=3, simulation_method=3)

    # s = Simulation(gases=[{'count': (i + 1) * 3, 'radius': j / 150}
    # for i, j in enumerate(range(16, 1, -3))], dimensions=3)

    #    s = Simulation(gases=[{'count': 50, 'radius': 0.1}], dimensions=3, animation=None)

    # s.run()
