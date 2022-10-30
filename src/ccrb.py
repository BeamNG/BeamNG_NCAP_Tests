from beamngpy import BeamNGpy
from beamng_ncap.scenarios import CCRB, generate_scenario

if __name__ == '__main__':
    beamng = BeamNGpy('localhost', 64256)
    beamng.open()
    scenario = generate_scenario('etk800', 'etk800')
    scenario.make(beamng)

    beamng.set_deterministic()
    beamng.set_steps_per_second(100)
    beamng.load_scenario(scenario)
    beamng.pause()
    beamng.start_scenario()

    print('Car-to-Car Rear braking scenario')
    deceleration = -6 # -2, -6
    distance = 40 # 12, 40
    overlap = 100  # -75, -50, 50, 75 or 100 %
    test = CCRB(beamng, deceleration, distance, overlap)

    # Add custom sensors here (electrics, damage & timer are already attached)
    # test.vut.attach_sensor(...)
    sensors = test.load()
    test_state = test.execute()

    if test_state == 1:
        print('Test passed successfully')
    elif test_state == -1:
        print('Test failed')
    else: 
        print('No terminal state reached') 
