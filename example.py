from beamngpy import BeamNGpy, Road, Scenario, Vehicle
from euro_ncap import CCRS, CCRB, CCRM, generate_scenario

bng = BeamNGpy('localhost', 64256, home='D:\\beamng\\research\\trunk')
bng.open()
scenario = generate_scenario("etk800", "etk800")
scenario.make(bng)
bng.set_deterministic()
bng.set_steps_per_second(100)
bng.load_scenario(scenario)
bng.pause()
bng.start_scenario()

### Car-to-Car Rear braking (4 possible Tests)
#deceleration = -2 # -2 or -6 m/s^2
#distance = 12 # 12 or 40 m
#test = CCRB(bng, deceleration, distance)
#sensors = test.load()

### Car-to-Car Rear moving (55 possible Tests)
#overlap = 100 # -75, -50, 50, 75 or 100 %
#vut_speed = 60 # 30, 35, 40, 45, 50, 55, 60, 65, 70, 75 or 80 km/h
#test = CCRM(bng, 65, overlap)
#sensors = test.load()

### Car-to-Car Rear stationary (45 possible Tests)
overlap = 100 # -75, -50, 50, 75 or 100 %
vut_speed = 45 # 10, 15, 20, 25, 30, 35, 40, 45 or 50 km/h
test = CCRS(bng, vut_speed, overlap)
# Add Sensors here (electrics, damage and timer are already attached)
#test.vut.attach_sensor(...)
sensors = test.load()
#state = 0
#while state == 0:
#TODO Add AEB/FCW logic here
#    test.vut.ai_set_mode("disabled")
#    test.vut.control(throttle=0, brake=0.5)
#    sensors = test.step(20) # 20 == 1s
#    test.get_state(sensors)

#if state == 1:
#    print("Success")
#else:
#    print("Failure")
