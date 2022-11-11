

class PID:
    '''
    PID controller class
    Args:
        Kp (float): proportional coefficient
        Ki (float): integral coefficient
        Kd (float): derivative coefficient
    '''

    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def actuation(self, e: float, dt: float) -> float:
        '''
        Actuate the PID controller
        Args:
            e (float): input error wrt the target
            dt (float): time step to coumpute integral and derivative
        Returns:
            output (float): output value of the control action
        '''
        self.integral += (e + self.prev_error)*dt/2
        derivative = (e - self.prev_error)/dt
        output = self.Kp*e + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = e

        return output


class TrialControl:
    '''
    Simple controller that aim to mantain at least a minimum time distance
    wrt to the vehicle in front
    Args:
        time_distance (float): minimum time distance allowable
        pid (PID): instance of the PID class
    '''
    def __init__(self, time_distance: float, pid: PID):
        self.time_distance = time_distance
        self.pid = pid

        assert self.pid.Kp > 0

    def get_target_distance(self, speed: float):
        '''
        Convert the time distance taget to a spatial distance
        Args:
            speed (float): vehicle speed [m/s]
        '''
        self.target_distance = speed*self.time_distance

    def actuation(self, actual_distance: float, dt: float) -> float:
        '''
        Actuate the PID controller using the spatial distance as target
        Args:
            actual_distance (float): distance between the two cars
            dt (float): time step to coumpute integral and derivative
        Returns:
            brake (float): amount of brake needed to follow the target
        '''
        error = self.target_distance - actual_distance
        brake = self.pid.actuation(error, dt)

        brake = 1 if brake > 1 else brake
        brake = 0 if brake < 0 else brake

        return brake
