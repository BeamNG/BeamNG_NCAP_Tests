
from beamngpy import BeamNGpy
from beamng_ncap.scenarios import CCRS, CCRM, generate_scenario

if __name__ == '__main__':
    beamng = BeamNGpy('localhost', 25252)
    beamng.open()
    scenario = generate_scenario('etk800', 'etk800')
    scenario.make(beamng)

    beamng.set_deterministic()
    beamng.set_steps_per_second(100)
    beamng.load_scenario(scenario)
    beamng.pause()
    beamng.start_scenario()

    print('Car-to-Car Rear stationary scenario')
    overlap = 100  # -75, -50, 50, 75 or 100 %
    vut_speed = 50  # 10, 15, 20, 25, 30, 35, 40, 45 or 50 km/h
    test = CCRS(beamng, vut_speed, overlap=overlap)

    sensors = test.load()
    
    print(f"Running test with VUT speed: {vut_speed} km/h")
    print("Using TRIAL mode for automated testing...")
    test_state, test_score = test.execute('trial')

    if test_state == 1:
        print('Test passed successfully')
    elif test_state == -1:
        print('Test failed')
    else:
        print('No terminal state reached')

    print(f'Score: {test_score}')
