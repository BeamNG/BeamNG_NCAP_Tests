"""
Scenarios for testing autonomous emergency braking (AEB), emergency lane
keeping (ELK) and lane keep assist (LKA) systems based on European New Car
Assessment Programme (Euro NCAP) test protocols.
References:
[1] European New Car Assessment Programme. Test Protocol - AEB Car-to-Car
    systems. Version 3.0.2. July 2019.
    https://cdn.euroncap.com/media/56143/euro-ncap-aeb-c2c-test-protocol-v302.pdf
[2] European New Car Assessment Programme. Test Protocol - AEB VRU systems.
    Version 3.0.2. July 2019.
    https://cdn.euroncap.com/media/53153/euro-ncap-aeb-vru-test-protocol-v302.pdf
[3] European New Car Assessment Programme. Assessment Protocol - Safety Assist.
    Version 9.1 November 2021
    https://cdn.euroncap.com/media/67254/euro-ncap-assessment-protocol-sa-v91.pdf

.. moduleauthor:: Sedonas <https://github.com/Sedonas>
.. moduleauthor:: Marc Müller <mmueller@beamng.gmbh>
.. moduleauthor:: Mattia Vicari <mvicari@beamng.gmbh>
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import time
import numpy as np
import keyboard
from beamngpy import BeamNGpy, Road, Scenario, ScenarioObject, Vehicle
from beamngpy.sensors import Damage, Electrics, Timer

from .controllers import PID, TrialControl

Pos = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]


def generate_scenario(vut_model: str, gvt_model: str) -> Scenario:
    """
    Generates the Scenario.
    Args:
        vut_model (str): Model of the VUT.
        gvt_model (str): Model of the GVT.
    """
    scenario = Scenario('smallgrid', 'ncap_scenario')
    road1 = Road('track_editor_C_center', rid='ncap_road1')
    road1.add_nodes((0, 5000, 0, 7), (0, -5000, 0, 7))
    scenario.add_road(road1)

    road2 = Road('track_editor_C_center', rid='ncap_road2', interpolate=False)
    road2.add_nodes((500, -1000, 0, 7), (500, 1000, 0, 7))
    scenario.add_road(road2)

    road3 = Road('track_editor_C_center', rid='ncap_road3', interpolate=False)
    road3.add_nodes((250, 0, 0, 7), (750, 0, 0, 7))
    scenario.add_road(road3)

    curve1 = Road('track_editor_C_center', rid='ncap_curve1',
                  interpolate=False)
    curve1.add_nodes(*_generate_curve(500, -11.5, 270, 178, -2, 11.5, 7))
    scenario.add_road(curve1)

    curve2 = Road('track_editor_C_center', rid='ncap_curve2',
                  interpolate=False)
    curve2.add_nodes(*_generate_curve(500, -11.5, 90, 182, 2, 11.5, 7))
    scenario.add_road(curve2)

    curve3 = Road('track_editor_C_center', rid='ncap_curve3',
                  interpolate=False)
    curve3.add_nodes(*_generate_curve(500, 11.5, 270, 362, 2, 11.5, 7))
    scenario.add_road(curve3)

    curve4 = Road('track_editor_C_center', rid='ncap_curve4',
                  interpolate=False)
    curve4.add_nodes(*_generate_curve(500, 11.5, 90, -2, -2, 11.5, 7))
    scenario.add_road(curve4)

    road4 = Road('track_editor_C_center', rid='ncap_road4', interpolate=False)
    road4.add_nodes((-500, 1000, 0, 7), (-500, -1000, 0, 7))
    scenario.add_road(road4)

    ccrs_waypoint = ScenarioObject('ccr_waypoint', None, 'BeamNGWaypoint',
                                   (0, -5000, 0.23), (1, 1, 1), (0, 0, 0, 1))
    scenario.add_object(ccrs_waypoint)

    ccftab_waypoint = ScenarioObject('ccftab_waypoint_gvt', None,
                                     'BeamNGWaypoint', (500, 1000, 0.23),
                                     (1, 1, 1), (0, 0, 1, 0))
    scenario.add_object(ccftab_waypoint)

    ccftab_waypoint2 = ScenarioObject('ccftab_waypoint_vut', None,
                                      'BeamNGWaypoint', (500, -1000, 0.23),
                                      (1, 1, 1), (0, 0, 0, 1))
    scenario.add_object(ccftab_waypoint2)

    vut = Vehicle('vut', model=vut_model, licence='VUT')
    vut.attach_sensor('electrics', Electrics())
    vut.attach_sensor('damage', Damage())
    vut.attach_sensor('timer', Timer())
    scenario.add_vehicle(vut, (500, 500, 0.23))

    gvt = Vehicle('gvt', model=gvt_model, licence='GVT')
    gvt.attach_sensor('electrics', Electrics())
    gvt.attach_sensor('damage', Damage())
    gvt.attach_sensor('timer', Timer())
    scenario.add_vehicle(gvt, (505, 500, 0.23))

    # TODO Add Pedestrian and Cyclist

    return scenario


def _generate_curve(x: float, y: float, angle_start: float,
                    angle_stop: float, angle_step: float, radius: float,
                    width: float) -> List[Tuple[float, float, float, float]]:
    x_off = x - radius * np.sin(np.radians(angle_start))
    y_off = y - radius * -np.cos(np.radians(angle_start))

    nodes = []
    for a in range(angle_start, angle_stop, angle_step):
        x = radius * np.sin(np.radians(a)) + x_off
        y = radius * -np.cos(np.radians(a)) + y_off
        nodes.append((x, y, 0, width))

    return nodes


class NCAPScenario(ABC):
    """
    Abstract base class for all NCAP scenarios.
    Args:
        bng (beamngpy.BeamNGpy): BeamNGpy instance.
        vut_speed (float): Speed of the VUT in km/h.
        vut_position (tuple): Spawn position of the VUT.
        vut_rotation (tuple): Spawn rotation quaternion of the VUT.
        vut_waypoint (str): Name of the waypoint.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, vut_position: Pos,
                 vut_rotation: Quat, vut_waypoint: str):
        self.bng = bng
        self.vut: Vehicle = bng.scenario.get_vehicle('vut')
        self._vut_speed = vut_speed / 3.6
        self._vut_position = vut_position
        self._vut_rotation = vut_rotation
        self._vut_waypoint = vut_waypoint

    @abstractmethod
    def load(self):
        """
        Loads the scenario.
        """
        pass

    def reset(self):
        """
        Resets the scenario.
        """
        self.bng.restart_scenario()

    @abstractmethod
    def step(self, steps):
        """
        Advances the scenario the given amount of steps.
        Args:
            steps (int): The amount of steps to simulate.
        """
        pass

    @abstractmethod
    def get_state(self, sensors):
        """
        Returns the state of the test.
        Args:
            sensors (dict): The sensor data from both vehicles as a dictionary
                            of dictionaries.
        """
        pass

    @abstractmethod
    def get_score(self, sensors):
        """
        Returns the score of the test.
        Args:
            sensors (dict): The sensor data from both vehicles as a dictionary
                            of dictionaries.
        Returns:
            score (int): final score of the test.
        """
        pass


class CCScenario(NCAPScenario):
    """
    Base class for the car to car scenarios.
    Args:
        bng (beamngpy.BeamNGpy): BeamNGpy instance.
        vut_speed (float): Speed of the VUT in km/h.
        vut_position (tuple): Spawn position of the VUT.
        vut_waypoint (str): Name of the waypoint.
        gvt_speed (float): Speed of the GVT in km/h.
        gvt_position (tuple): Spawn position of the VUT.
        gvt_waypoint (str): Name of the waypoint.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, vut_position: Pos,
                 vut_rotation: Quat, vut_waypoint: str, gvt_speed: float,
                 gvt_position: Pos, gvt_rotation: Quat, gvt_waypoint: str):
        super().__init__(bng, vut_speed, vut_position, vut_rotation,
                         vut_waypoint)

        self._vut_speed_ai = (vut_speed + 3.6 + 1)/3.6
        # + 3.6 ) / is a workaround for the ai to reach the given speed
        # TODO: investigate how to get the AI to do that without this hack

        self.gvt: Vehicle = bng.scenario.get_vehicle('gvt')
        self._gvt_speed = gvt_speed / 3.6
        self._gvt_speed_ai = (gvt_speed + 3.6) / 3.6
        self._gvt_position = gvt_position
        self._gvt_rotation = gvt_rotation
        self._gvt_waypoint = gvt_waypoint

    def _teleport_vehicle(self, vehicle: Vehicle, position: Pos,
                          rotation: Quat | None = None):
        """
        Teleports the given vehicle to the given position with the given
        rotation.
        Args:
            vehicle (beamngpy.Vehicle): The vehicle to teleport.
            position (tuple) The target position as an (x, y, z) tuple
                             containing world-space coordinates.
            rotation (tuple): Optional tuple specifying the rotation as
                              a quaternion.
        Notes:
            This method is a workaround that ignores the refnode of the vehicle
            and uses the center of the boundingbox instead.
        """
        vehicle.teleport(position, rotation)
        position = np.array(position)

        center = np.zeros(3)
        for value in vehicle.get_bbox().values():
            center += np.array(value)

        offset = position - center / 8
        offset[2] = 0  # Ignore z coordinate

        vehicle.teleport(list(position + offset), rotation)

    def _observe(self) -> Dict[str, Dict]:
        """
        Retrieves sensor values for the sensors attached to the VUT and GVT and
        returns them as a dict.
        Returns:
            The sensor data from both vehicles as a dictionary of dictionaries.
        Notes:
            Example: observation['vut']['electrics']['wheelspeed']
        """
        observation = dict()
        for vehicle in [self.vut, self.gvt]:
            vehicle.poll_sensors()
            sensors = dict()

            for key, value in vehicle.sensors.items():
                sensors[key] = value.data

            observation[vehicle.vid] = sensors

        return observation

    def _get_speeds(self, poll=True) -> tuple[float, float]:
        """
        Returns the velocities of the vut and gvt vehicles from the state attribute.
        Args:
            poll (bool): if true re-poll data.
        Returns:
            vut_speed (float): vut_speed in m/s.
            gvt_speed (float): gvt_speed in m/s.
        """
        if poll:
            self.vut.poll_sensors()
            self.gvt.poll_sensors()

        vut_speed = np.sqrt((self.vut.state['vel'][0])**2
                            + (self.vut.state['vel'][1])**2
                            + (self.vut.state['vel'][2])**2)
        gvt_speed = np.sqrt((self.gvt.state['vel'][0])**2
                            + (self.gvt.state['vel'][1])**2
                            + (self.gvt.state['vel'][2])**2)

        return vut_speed, gvt_speed

    def _accelerate_cars(self):
        """
        Accelerates the VUT and GVT to the desired speed.
        """
        vut_speed = 0
        gvt_speed = 0

        while not self._cars_reached_speed(vut_speed, gvt_speed) or self._ttc() > 4:
            self.bng.step(10)
            vut_speed, gvt_speed = self._get_speeds()

    def _cars_reached_speed(self, vut_speed: float, gvt_speed: float):
        """
        Checks whether both cars have reached the desired speed.
        Notes:
            See [1], page 20, section 8.4.2
        """
        vut = np.isclose(vut_speed, self._vut_speed+0.5/3.6, atol=0.5/3.6)
        gvt = np.isclose(gvt_speed, self._gvt_speed, atol=1/3.6)

        return vut and gvt

    def get_state(self, sensors: Dict[str, Dict]) -> int:
        """
        Returns the state of the test.
        Args:
            sensors (dict): The sensor data from both vehicles as a dictionary
            of dictionaries.
        Returns:
            0 = Non terminal state
            1 = Success (V_VUT = 0km/h or V_VUT < V_GVT)
           -1 = Failure (Contact between VUT and GVT)
        Notes:
            See [1], page 21, section 8.4.3
        """
        vut_dmg = sensors['vut']['damage']['damage']
        gvt_dmg = sensors['gvt']['damage']['damage']
        vut_speed, gvt_speed = self._get_speeds(poll=False)

        if vut_dmg != 0 or gvt_dmg != 0:
            return -1

        if np.isclose(vut_speed, 0, atol=1e-1) or vut_speed < gvt_speed:
            return 1

        return 0

    def _countdown(self, seconds: int):
        self.bng.pause()
        self.bng.display_gui_message('When the game restarts you can take '
                                     + 'control of the car pressing SPACE')
        time.sleep(3)
        for s in range(seconds, 0, -1):
            self.bng.display_gui_message('Get ready! The game will restart '
                                         + f'in {s} seconds.')
            time.sleep(1)
        self.bng.display_gui_message(f'Go!')
        self.bng.resume()


class CCRScenario(CCScenario):
    """
    Base class for the car to car rear scenarios.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, gvt_speed: float,
                 distance: float, overlap: float = 1.0):
        """
        Args:
            bng (:class:`.BeamNGpy`): BeamNGpy instance.
            vut_speed (float): Speed of the VUT in km/h.
            gvt_speed (float): Speed of the GVT in km/h.
            distance (float): Distance between the Cars in m.
            overlap (float): Percentage of the width of the VUT overlapping
                             the GVT.
        """
        super().__init__(bng, vut_speed, (0, 0, 0), (0, 0, 0, 1),
                         'ccr_waypoint', gvt_speed, (0, -1000, 0.21),
                         (0, 0, 0, 1), 'ccr_waypoint')
        self._overlap = overlap / 100
        self._initial_distance = 100
        self._distance = distance

    def load(self):
        """
        Loads the Scenario.
        """
        self.bng.switch_vehicle(self.vut)

        self._teleport_vehicle(self.gvt, self._gvt_position,
                               self._gvt_rotation)
        self._teleport_vut()

        self.vut.ai_set_speed(self._vut_speed_ai, mode='set')
        self.vut.ai_set_waypoint(self._vut_waypoint)

        if self._gvt_speed > 0:
            self.gvt.ai_set_speed(self._gvt_speed_ai, mode='set')
            self.gvt.ai_set_waypoint(self._gvt_waypoint)

        self._accelerate_cars()
        self._fix_boundary_conditions()

        return self._observe()

    def execute(self, control_mode='user') -> int:
        """
        Execute the test stopping it according to [1] section 8.4.3 pag 21.
        Args:
            control_mode (string): the control mode to use during the test
                                   execution.
                * ``user``: The user has to control the vehicle during
                            the test execution.
                * ``trial``: The test is performed using TrialControl.
        Returns:
            terminal state:
                * 1: Test passed successfully
                * -1: Test failed
                * 0: No terminal state reached
        """
        exit_condition1 = False
        exit_condition2 = False
        exit_condition3 = False
        self.boundary_conditions = {'vut_speed': [],
                                    'gvt_speed': [],
                                    'vut_x': [],
                                    'vut_y': [],
                                    'gvt_x': [],
                                    'gvt_y': [],
                                    'relative_distance': [],
                                    'vut_yaw_velocity': [],
                                    'gvt_yaw_velocity': [],
                                    'vut_steering': []}

        if control_mode == 'user':
            self._countdown(3)
            self.bng.pause()
            ai_disabled = False

            while not any([exit_condition1, exit_condition2, exit_condition3]):
                if keyboard.is_pressed('Space') and not ai_disabled:
                    self.vut.ai_set_mode('disabled')
                    self.bng.display_gui_message('AI disabled, now '
                                                 + 'you can control the car')
                    self.bng.resume()
                    ai_disabled = True
                elif not ai_disabled:
                    # need 1 step to compute precisely the impact speed
                    self.step(1)
                    sensors = self._observe()
                    vut_speed, gvt_speed = self._get_speeds(poll=False)
                    vut_x = self.vut.state['pos'][0]
                    vut_y = self.vut.state['pos'][1]
                    gvt_x = self.gvt.state['pos'][0]
                    gvt_y = self.gvt.state['pos'][1]
                    vut_steering = sensors['vut']['electrics']['steering']
                    self.boundary_conditions['vut_speed'].append(vut_speed)
                    self.boundary_conditions['gvt_speed'].append(gvt_speed)
                    self.boundary_conditions['vut_x'].append(vut_x)
                    self.boundary_conditions['vut_y'].append(vut_y)
                    self.boundary_conditions['gvt_x'].append(gvt_x)
                    self.boundary_conditions['gvt_y'].append(gvt_y)
                    self.boundary_conditions['vut_steering'].append(vut_steering)

                sensors = self._observe()
                vut_speed, gvt_speed = self._get_speeds()
                vut_dmg = sensors['vut']['damage']['damage']
                gvt_dmg = sensors['gvt']['damage']['damage']

                if np.isclose(vut_speed, 0, atol=1e-1):
                    exit_condition1 = True
                    self.bng.pause()
                elif vut_speed < gvt_speed:
                    exit_condition2 = True
                    self.bng.pause()
                elif vut_dmg or gvt_dmg:
                    exit_condition3 = True
                    self.bng.pause()

        else:
            controllers_dict = {'trial': self._get_trial_controller}
            actuation_dict = {'trial': self._actuate_trial_controller}
            ctrl = controllers_dict[control_mode]()

            while not any([exit_condition1, exit_condition2, exit_condition3]):
                self.step(10)

                steering, throttle, brake = actuation_dict[control_mode](ctrl)

                if any([steering, throttle, brake]):
                    self.vut.ai_set_mode('disabled')
                    self.vut.control(steering=steering,
                                     throttle=throttle,
                                     brake=brake)

                sensors = self._observe()
                vut_speed, gvt_speed = self._get_speeds()
                vut_dmg = sensors['vut']['damage']['damage']
                gvt_dmg = sensors['gvt']['damage']['damage']

                if np.isclose(vut_speed, 0, atol=1e-2):
                    exit_condition1 = True
                elif vut_speed < gvt_speed:
                    exit_condition2 = True
                elif vut_dmg or gvt_dmg:
                    exit_condition3

        self._check_boundary_conditions()
        score = self.get_score(sensors)
        state = self.get_state(sensors)

        return state, score

    def _check_boundary_conditions(self) -> None:
        """
        Check if all the bounday conditions described in [1] section 8.4.2 pag 20 are met.
        Print only if one or more conditions aren't respected.
        """
        vut_speed = True
        gvt_speed = True
        vut_path = True
        gvt_path = True
        for i in range(len(self.boundary_conditions['vut_speed'])):
            vut_speed = np.isclose(self.boundary_conditions['vut_speed'][i], self._vut_speed+0.5/3.6, atol=0.5/3.6) and vut_speed
            gvt_speed = np.isclose(self.boundary_conditions['gvt_speed'][i], self._gvt_speed, atol=1/3.6) and gvt_speed
            vut_path = np.isclose(self.boundary_conditions['vut_x'][i], self._vut_position[0], atol=0.05) and vut_path
            gvt_path = np.isclose(self.boundary_conditions['gvt_x'][i], self._gvt_position[0], atol=0.1) and gvt_path

            # TODO add others boundary conditions
            # TODO investigate why both vut and gvt x positions are circa -0.32

        if not vut_speed:
            print('VUT speed boundary condition not respected')
        if not gvt_speed:
            print('GVT speed boundary condition not respected')
        if not vut_path:
            print('VUT deviation from path boundary condition not respected')
        if not gvt_path:
            print('GVT deviation from path boundary condition not respected')

    def _get_distance(self):
        """
        Calculates the distance between VUT and GVT.
        """
        vut_bbox = self.vut.get_bbox()
        vut_c_a = np.array(vut_bbox['front_bottom_left'])
        vut_c_b = np.array(vut_bbox['front_bottom_right'])
        vut_c = (vut_c_a + vut_c_b) / 2
        vut_c[2] = 0  # Ignore Z

        gvt_bbox = self.gvt.get_bbox()
        gvt_p = np.array(gvt_bbox['rear_bottom_left'])
        gvt_p[2] = 0  # Ignore Z
        gvt_dv = np.array(gvt_bbox['rear_bottom_right']) - gvt_p
        gvt_dv[2] = 0  # Ignore Z

        return np.linalg.norm(np.cross(vut_c - gvt_p, gvt_dv)) / \
            np.linalg.norm(gvt_dv)

    def _ttc(self):
        """
        Evaluate the time to collision
        Returns:
            ttc (float): time to collision in seconds.
        """
        distance = self._get_distance()
        sensors = self._observe()
        vut_speed = sensors['vut']['electrics']['wheelspeed']
        gvt_speed = sensors['gvt']['electrics']['wheelspeed']
        relative_speed = vut_speed - gvt_speed
        relative_speed = 1 if relative_speed == 0 else relative_speed

        return distance/relative_speed


    def step(self, steps):
        """
        Advances the scenario the given amount of steps.
        Args:
            steps (int): The amount of steps to simulate.
        Returns:
            A Dictionary with the sensor data from the VUT and GVT.
        """
        self.bng.step(steps)
        observation = self._observe()

        return observation

    def _teleport_vut(self):
        """
        Spawns the VUT 'self.initial_distance' meters behind the GVT with an
        overlap of 'self.overlap'.
        """
        bbox_gvt = self.gvt.get_bbox()
        fbl = np.array(bbox_gvt['front_bottom_left'])
        fbr = np.array(bbox_gvt['front_bottom_right'])
        rbl = np.array(bbox_gvt['rear_bottom_left'])
        rbr = np.array(bbox_gvt['rear_bottom_right'])

        pct = np.abs(self._overlap) - 0.5

        if self._overlap < 0:
            point_rear = pct * (rbr - rbl) + rbl
            point_front = pct * (fbr - fbl) + fbl
        else:
            point_rear = pct * (rbl - rbr) + rbr
            point_front = pct * (fbl - fbr) + fbr

        dv = point_rear - point_front
        dv /= np.linalg.norm(dv)

        position = point_front + self._initial_distance * dv

        # sensors have to be polled once so the vehicle state gets populated
        self.gvt.poll_sensors()
        position[2] = self.gvt.state['pos'][2]

        self.vut.teleport(list(position), self._vut_rotation)

        vut_bbox = self.vut.get_bbox()
        vut_fc_a = np.array(vut_bbox['front_bottom_left'])
        vut_fc_b = np.array(vut_bbox['front_bottom_right'])
        vut_fc = (vut_fc_a + vut_fc_b) / 2

        offset = position - vut_fc
        offset[2] = 0  # Ignore z coordinate

        self.vut.teleport(list(position + offset), self._vut_rotation)

    def _get_trial_controller(self):
        """
        Define the TrialControl used to perform the test
        Returns:
            instance of the contoller TrialControl
        """
        pid = PID(0.5, 0, 0)
        controller = TrialControl(1.5, pid)

        return controller

    def _actuate_trial_controller(self, controller: TrialControl) -> tuple:
        """
        Actuation routing for the TrialControl.
        Args:
            controller (TrialControl): instance of the controller
        Returns:
            tuple containing sterring, throttle and brake
        """
        observation = self._observe()
        vut_speed = observation['vut']['electrics']['wheelspeed']
        controller.get_target_distance(vut_speed)
        actual_distance = self._get_distance()
        brake = controller.actuation(actual_distance, 0.1)
        throttle = 0
        steering = 0

        return steering, throttle, brake

    def get_score(self, sensors):
        """
        Returns the score of the test.
        Args:
            sensors (dict): The sensor data from both vehicles as a dictionary
                            of dictionaries.
        Returns:
            score (int): final score of the test.
        """
        vut_speed, gvt_speed = self._get_speeds(poll=False)
        impact_relative_speed = (vut_speed - gvt_speed)*3.6
        color_scheme = self._get_color_scheme()
        score = 0

        for i, treshold in enumerate(color_scheme['tresholds']):
            if impact_relative_speed < treshold:
                score = color_scheme['scores'][i]

        return score

    def _fix_boundary_conditions(self):
        """
        Fix the boundary conditions according to [1] section 8.4.2 pag 20
        """
        pass

    def _get_color_scheme(self):
        """
        Select the right color scheme according to the vut speed.
        See [1] section 6.1.3 pag 9.
        """
        pass


class CCRS(CCRScenario):
    """
    Implementation of the Car-to-Car Rear stationary scenario.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, overlap: int):
        """
        Instantiates the Scenario.
        Args:
            bng (beamng.BeamNGpy): BeamNGpy instance.
            vut_speed (float): Speed of the VUT in km/h.
            overlap (int): Percentage of the width of the VUT overlapping
                           the GVT.
        """
        assert vut_speed in range(10, 55, 5)
        assert overlap in [-75, -50, 50, 75, 100]

        distance = vut_speed / 3.6 * 4
        super(CCRS, self).__init__(bng, vut_speed, 0, distance, overlap)

    def _get_color_scheme(self):
        """
        Select the right color scheme according to the vut speed.
        See [1] section 6.1.3 pag 9.
        """
        color_scheme_50 = {'tresholds': [40, 30, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_45 = {'tresholds': [35, 25, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_40 = {'tresholds': [35, 25, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_35 = {'tresholds': [25, 15, 5],
                           'scores': [0.5, 0.75, 1]}
        color_scheme_30 = {'tresholds': [25, 15, 5],
                           'scores': [0.5, 0.75, 1]}
        color_scheme_25 = {'tresholds': [15, 5],
                           'scores': [0.5, 1]}
        color_scheme_20 = {'tresholds': [1],
                           'scores': [1]}
        color_scheme_15 = {'tresholds': [1],
                           'scores': [1]}
        color_scheme_10 = {'tresholds': [1],
                           'scores': [1]}

        color_schemes = {'50': color_scheme_50,
                         '45': color_scheme_45,
                         '40': color_scheme_40,
                         '35': color_scheme_35,
                         '30': color_scheme_30,
                         '25': color_scheme_25,
                         '20': color_scheme_20,
                         '15': color_scheme_15,
                         '10': color_scheme_10}

        return color_schemes[str(int(self._vut_speed*3.6))]


class CCRM(CCRScenario):
    """
    Implementation of the Car-to-Car Rear moving scenario.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, overlap: int):
        """
        Instantiates the Scenario.
        Args:
            bng (beamng.BeamNGpy): BeamNGpy instance.
            vut_speed (float): Speed of the VUT in km/h.
            overlap (int): Percentage of the width of the VUT overlapping the
                           GVT.
        """
        assert vut_speed in range(30, 85, 5)
        assert overlap in [-75, -50, 50, 75, 100]

        distance = vut_speed / 3.6 * 4 - 20 / 3.6 * 4
        super(CCRM, self).__init__(bng, vut_speed, 20, distance, overlap)

    def _get_color_scheme(self):
        """
        Select the right color scheme according to the vut speed.
        See [1] section 6.1.3 pag 10.
        """
        color_scheme_80 = {'tresholds': [50, 35, 20, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_75 = {'tresholds': [45, 30, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_70 = {'tresholds': [40, 30, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_65 = {'tresholds': [35, 25, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_60 = {'tresholds': [35, 25, 15, 5],
                           'scores': [0.25, 0.5, 0.75, 1]}
        color_scheme_55 = {'tresholds': [25, 15, 5],
                           'scores': [0.5, 0.75, 1]}
        color_scheme_50 = {'tresholds': [25, 15, 5],
                           'scores': [0.5, 0.75, 1]}
        color_scheme_45 = {'tresholds': [15, 5],
                           'scores': [0.5, 1]}
        color_scheme_40 = {'tresholds': [15, 5],
                           'scores': [0.5, 1]}
        color_scheme_35 = {'tresholds': [5],
                           'scores': [1]}
        color_scheme_30 = {'tresholds': [5],
                           'scores': [1]}

        color_schemes = {'80': color_scheme_80,
                         '75': color_scheme_75,
                         '70': color_scheme_70,
                         '65': color_scheme_65,
                         '60': color_scheme_60,
                         '55': color_scheme_55,
                         '50': color_scheme_50,
                         '45': color_scheme_45,
                         '40': color_scheme_40,
                         '35': color_scheme_35,
                         '30': color_scheme_30}

        return color_schemes[str(int(self._vut_speed*3.6))]


class CCRB(CCRScenario):
    """
    Implementation of the Car-to-Car Rear braking scenario.
    """

    def __init__(self, bng, deceleration, distance, overlap=100):
        """
        Instantiates the Scenario.
        Args:
            bng (beamng.BeamNGpy): BeamNGpy instance.
            deceleration (float): Deceleration of the GVT in m/s^2.
            distance (float): Distance between the Cars in m.
            overlap (int): Percentage of the width of the VUT overlapping
                           the GVT.
        """
        assert deceleration in [-2, -6]
        assert distance in [12, 40]

        super(CCRB, self).__init__(bng, 50, 50, distance, overlap)
        self._deceleration = - deceleration
        self._decelerating = False
        self._stationary = False
        self._gvt_controller = PID(0.2, 0.5, 0)

    def reset(self):
        """
        Resets the scenario.
        """
        self._decelerating = False
        super().reset()

    def _accelerate_cars(self):
        """
        Accelerates the VUT and GVT to the desired speed.
        """
        vut_speed = 0
        gvt_speed = 0

        while not self._cars_reached_speed(vut_speed, gvt_speed):
            self.bng.step(10)
            observation = self._observe()
            vut_speed = observation['vut']['electrics']['wheelspeed']
            gvt_speed = observation['gvt']['electrics']['wheelspeed']

    def step(self, steps):
        """
        Advances the scenario the given amount of steps.
        Args:
            steps (int): The amount of steps to simulate.
        Returns:
            A Dictionary with the sensor data from the VUT and GVT.
        """
        if not self._decelerating:
            self.gvt.ai_set_mode('disabled')
            self._decelerating = True

        if self._decelerating and not self._stationary:
            sensors = self._observe()
            gvt_acc = sensors['gvt']['electrics']['accYSmooth']
            error = self._deceleration - gvt_acc
            brake = self._gvt_controller.actuation(error, 0.1)
            brake = 1 if brake > 1 else brake
            brake = 0 if brake < 0 else brake
            self.gvt.control(throttle=0, brake=brake)
            gvt_speed = sensors['gvt']['electrics']['wheelspeed']

            if np.isclose(gvt_speed, 0, atol=1e-1):
                self.gvt.control(throttle=0, brake=0)
                self._stationary = True

        return super().step(steps)

    def _get_color_scheme(self):
        """
        Select the right color scheme according to the vut speed.
        See [1] section 6.1.3 pag 10.
        """
        color_scheme = {'tresholds': [1],
                        'scores': [1]}

        return color_scheme

    def _fix_boundary_conditions(self):
        """
        Fix the relative distance between VUT and GVT according to [1]
        section 8.4.2 pag 20.
        """
        dist_diff = self._get_distance() - self._distance

        if not np.isclose(dist_diff, 0, atol=0.5):
            vut_bbox = self.vut.get_bbox()
            vut_cfg_a = np.array(vut_bbox['front_bottom_left'])
            vut_cfg_b = np.array(vut_bbox['front_bottom_right'])
            vut_cf = (vut_cfg_a + vut_cfg_b) / 2

            vut_cr_a = np.array(vut_bbox['rear_bottom_left'])
            vut_cr_b = np.array(vut_bbox['rear_bottom_right'])
            vut_cr = (vut_cr_a + vut_cr_b) / 2

            if dist_diff < 0:
                dv = vut_cr - vut_cf
            else:
                dv = vut_cf - vut_cr

            dv /= np.linalg.norm(dv)

            position = vut_cf + np.abs(dist_diff) * dv
            position[2] = self.vut.state['pos'][2]

            self.vut.teleport(list(position), reset=False)


class CCFScenario(CCScenario):
    """
    Base class for the car to car front scenarios.
    """

    def __init__(self, bng: BeamNGpy, vut_speed: float, gvt_speed: float):
        """
        Args:
            bng (:class:`.BeamNGpy`): BeamNGpy instance.
            vut_speed (float): Speed of the VUT in km/h.
            gvt_speed (float): Speed of the GVT in km/h.
            distance (float): Distance between the Cars in m.
        """
        super().__init__(bng, vut_speed,
                         (500 - 1.75, self._vut_start_y, 0.21),
                         (0, 0, 0, 1), 'ccftab_waypoint_vut',
                         gvt_speed, (500 + 1.75, self._gvt_start_y, 0.21),
                         (0, 0, 1, 0), 'ccftab_waypoint_gvt')

    def _follow_trajectories(self):
        """
        Make the AIs follow the given trajectories until ttc is <= 4.
        """
        vut_script = self._define_vut_trajectory()
        gvt_script = self._define_gvt_trajectory()
        self.vut.ai_set_script(vut_script)
        self.gvt.ai_set_script(gvt_script)

        while self._ttc() > 4:
            self.bng.step(10)

    def _ttc(self) -> float:
        """
        Evaluate the time to collision
        Returns:
            ttc (float): time to collision in seconds.
        """
        vut_bbox = self.vut.get_bbox()
        vut_c_a = np.array(vut_bbox['front_bottom_left'])
        vut_c_b = np.array(vut_bbox['front_bottom_right'])
        vut_c = (vut_c_a + vut_c_b) / 2
        vut_c[2] = 0  # Ignore Z

        gvt_bbox = self.gvt.get_bbox()
        gvt_c_a = np.array(gvt_bbox['front_bottom_left'])
        gvt_c_b = np.array(gvt_bbox['front_bottom_right'])
        gvt_c = (gvt_c_a + gvt_c_b)/2
        gvt_c[2] = 0  # Ignore Z

        distance = np.linalg.norm(vut_c - gvt_c)

        sensors = self._observe()
        vut_speed = sensors['vut']['electrics']['wheelspeed']
        gvt_speed = sensors['gvt']['electrics']['wheelspeed']
        relative_speed = vut_speed + gvt_speed
        relative_speed = 1 if relative_speed == 0 else relative_speed

        return distance/relative_speed

    def load(self):
        """
        Loads the Scenario.
        """
        self._teleport_vehicle(self.gvt, self._gvt_position,
                               self._gvt_rotation)
        self._teleport_vehicle(self.vut, self._vut_position,
                               self._vut_rotation)

        self.bng.switch_vehicle(self.vut)

        self._follow_trajectories()

        return self._observe()

    def execute(self, control_mode='user') -> int:
        """
        Execute the test stopping it according to [1] section 8.4.3 pag 21
        Args:
            control_mode (string): the control mode to use during the test
                                   execution.
                * ``user``: The user has to control the vehicle during the
                            test execution.
        Returns:
            terminal state:
                * 1: Test passed successfully
                * -1: Test failed
                * 0: No terminal state reached
        """
        exit_condition1 = False
        exit_condition3 = False

        if control_mode == 'user':
            self._countdown(3)
            self.bng.pause()
            ai_disabled = False

            while not any([exit_condition1, exit_condition3]):
                if keyboard.is_pressed('Space') and not ai_disabled:
                    self.vut.ai_set_mode('disabled')
                    self.bng.display_gui_message('AI disabled,' +
                                                 'now you can control the car')
                    self.bng.resume()
                    ai_disabled = True
                elif not ai_disabled:
                    self.step(10)

                sensors = self._observe()
                vut_dmg = sensors['vut']['damage']['damage']
                vut_speed = sensors['vut']['electrics']['wheelspeed']
                gvt_dmg = sensors['gvt']['damage']['damage']

                if np.isclose(vut_speed, 0, atol=1e-2):
                    exit_condition1 = True
                    self.bng.pause()
                elif vut_dmg or gvt_dmg:
                    exit_condition3 = True
                    self.bng.pause()

        state = self.get_state(sensors)
        score = self.get_score(state)

        return state, score

    def step(self, steps):
        """
        Advances the scenario the given amount of steps.
        Args:
            steps (int): The amount of steps to simulate.
        Returns:
            A Dictionary with the sensor data from the VUT and GVT.
        """
        self.bng.step(steps)
        observation = self._observe()

        return observation


class CCFTAP(CCFScenario):
    """
    Implementation of the Car-to-Car Front turn-across-path
    """
    def __init__(self, bng: BeamNGpy, vut_speed: float, gvt_speed: float):
        """
        Args:
            bng (:class:`.BeamNGpy`): BeamNGpy instance.
            vut_speed (float): Speed of the VUT in km/h.
            gvt_speed (float): Speed of the GVT in km/h.
            distance (float): Distance between the Cars in m.
        """

        assert vut_speed in range(10, 25, 5)
        assert gvt_speed in [30, 45, 55]

        # TODO better trajectories synchronisation needed
        self._vut_start_y = 40
        self._gvt_start_y = - int((self._vut_start_y
                                   - (gvt_speed - vut_speed)
                                   * vut_speed / gvt_speed)
                                  / vut_speed * gvt_speed)

        super(CCFTAP, self).__init__(bng, vut_speed, gvt_speed)

    def _define_vut_trajectory(self, debug: bool = False) -> list:
        """
        Define the trajectory for the vut vehicle
        Args:
            debug (bool): if `True` display the trajectory.
        Returns:
            script (list): list containing the nodes described as dictionary
                           with the fields: 'x', 'y', 'z' amnd 't'.
        """
        match self._vut_speed*3.6:
            case 10:
                alpha = np.deg2rad(20.62)
                beta = np.deg2rad(48.76)
                r2 = 9
            case 15:
                alpha = np.deg2rad(20.93)
                beta = np.deg2rad(48.14)
                r2 = 11.75
            case 20:
                alpha = np.deg2rad(21.79)
                beta = np.deg2rad(46.42)
                r2 = 14.75

        L = 2*alpha*r2
        x_clothoid = []
        y_clothoid = []

        for i in range(int(L)):
            x_clothoid.append(i - i**5/(40*(r2*L)**2)
                              + i**9/(3465*(r2*L)**4)
                              - i**13/(599040*(r2*L)**6))
            y_clothoid.append(i**3/(6*(r2*L))
                              - i**7/(336*(r2*L)**3)
                              + i**11/(42240*(r2*L)**5)
                              - i**15/(9676800*(r2*L)**7))

        x_part2 = 2*r2*np.sin(beta/2)/np.sqrt(2)
        y_clothoid1 = - 1.75 + x_clothoid[-1] + y_clothoid[-1] + x_part2
        y_constant_radius = - 1.75 + y_clothoid[-1] + x_part2

        script = []
        points = []
        point_color = [0, 0, 0, 0.1]
        sphere_coordinates = []
        sphere_radii = []
        sphere_colors = []

        clothoid_index1 = 0
        constant_radius_index = 1
        constant_radius_segment = 2*r2*np.sin(1/r2/2)
        y = self._vut_position[1]
        i = 0

        while y >= - 1.75:
            i += 1
            t = i/self._vut_speed
            if y >= y_clothoid1:
                node = {'x': self._vut_position[0],
                        'y': y,
                        'z': 0.21,
                        't': t}
                script.append(node)
                points.append([node['x'], node['y'], node['z']])
                y -= 1

            elif y < y_clothoid1 and y > y_constant_radius:
                if clothoid_index1 == 0:
                    start_y = script[-1]['y']
                delta_y = script[-1]['y'] - (start_y
                                             - x_clothoid[clothoid_index1])
                y -= delta_y
                node = {'x': self._vut_position[0]
                        + y_clothoid[clothoid_index1],
                        'y': start_y - x_clothoid[clothoid_index1],
                        'z': 0.21,
                        't': t}
                script.append(node)
                points.append([node['x'], node['y'], node['z']])
                clothoid_index1 += 1

            elif y <= y_constant_radius:
                gamma_k = np.pi/4 - beta/2 + 1/r2*constant_radius_index/2
                delta_x = constant_radius_segment*np.sin(gamma_k)
                delta_y = constant_radius_segment*np.cos(gamma_k)

                node = {'x': script[-1]['x'] + delta_x,
                        'y': script[-1]['y'] - delta_y,
                        'z': 0.21,
                        't': t}
                script.append(node)
                points.append([node['x'], node['y'], node['z']])

                constant_radius_index += 2
                y -= delta_y

            if y <= y_clothoid[-1] - 1.75:
                start_x = script[-1]['x']
                start_y = script[-1]['y']
                for j in range(100):
                    i += 1
                    t = i/self._vut_speed

                    if j < x_clothoid[-1]:
                        node = {'x': start_x + x_clothoid[-1]
                                - x_clothoid[-j-1],
                                'y': start_y - y_clothoid[-1]
                                + y_clothoid[-j-1],
                                'z': 0.21,
                                't': t}
                        script.append(node)
                        points.append([node['x'], node['y'], node['z']])

                    else:
                        node = {'x': self._vut_position[0] + y_clothoid[-1]
                                + x_part2 + j,
                                'y': script[-1]['y'],  # TODO y != -1.75
                                'z': 0.21,
                                't': t}
                        script.append(node)
                        points.append([node['x'], node['y'], node['z']])

                    sphere_coordinates.append([node['x'], node['y'],
                                              node['z']])
                    sphere_radii.append(0.1)
                    sphere_colors.append([1, 0, 0, 0.8])

                break

            sphere_coordinates.append([node['x'], node['y'], node['z']])
            sphere_radii.append(0.1)
            sphere_colors.append([1, 0, 0, 0.8])

        if debug:
            self.bng.add_debug_spheres(sphere_coordinates, sphere_radii,
                                       sphere_colors, cling=True, offset=0.1)
            self.bng.add_debug_polyline(points, point_color,
                                        cling=True, offset=0.1)

        return script

    def _define_gvt_trajectory(self, debug: bool = False) -> list:
        """
        Define the trajectory for the gvt vehicle
        Args:
            debug (bool): if `True` display the trajectory.
        Returns:
            script (list): list containing the nodes described as dictionary
                           with the fields: 'x', 'y', 'z' amnd 't'.
        """
        script = []
        points = []
        point_color = [0, 0, 0, 0.1]
        sphere_coordinates = []
        sphere_radii = []
        sphere_colors = []
        i = 0

        for y in range(self._gvt_position[1], 1000):
            i += 1
            t = i/self._gvt_speed
            node = {'x': self._gvt_position[0],
                    'y': y,
                    'z': 0.21,
                    't': t}
            script.append(node)
            points.append([node['x'], node['y'], node['z']])

            sphere_coordinates.append([node['x'], node['y'], node['z']])
            sphere_radii.append(0.1)
            sphere_colors.append([1, 0, 0, 0.8])

        if debug:
            self.bng.add_debug_spheres(sphere_coordinates, sphere_radii,
                                       sphere_colors, cling=True, offset=0.1)
            self.bng.add_debug_polyline(points, point_color,
                                        cling=True, offset=0.1)

        return script

    def get_score(self, state) -> int:
        """
        Returns the score of the test.
        Args:
            state (int): final state of the test.
        Returns:
            score (int): final score of the test.
        """
        score = 1 if state == 1 else 0

        return score
