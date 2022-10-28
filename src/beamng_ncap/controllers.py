import numpy as np

class PID:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def actuation(self, e: float, dt: float) -> float:
        self.integral += (e + self.prev_error)*dt/2
        derivative = (e - self.prev_error)/dt
        output = self.Kp*e + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = e

        return output


class SafeDistanceControl:
    def __init__(self, time_distance: float, pid: PID):
        self.time_distance = time_distance
        self.pid = pid

        assert self.pid.Kp > 0

    def get_target_distance(self, speed: float):
        self.target_distance = speed*self.time_distance

    def actuation(self, actual_distance: float, dt: float) -> float:
        error = self.target_distance - actual_distance
        brake = self.pid.actuation(error, dt)

        brake = 1 if brake > 1 else brake
        brake = 0 if brake < 0 else brake

        return brake
