from beamngpy import BeamNGpy
from beamng_ncap.scenarios import CCRB, generate_scenario

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

    print('Euro NCAP AEB Car-to-Car Rear Braking (CCRb) Test')
    deceleration = -6  # -2, -6
    distance = 12  # 12, 40
    overlap = 100  # -75, -50, 50, 75 or 100 %
    test = CCRB(beamng, deceleration, distance, overlap)

    sensors = test.load()
    
    print(f"Running test with deceleration: {deceleration} m/s², distance: {distance} m")
    print("Using TRIAL mode for automated testing...")
    test_state, test_score = test.execute('trial')

    if test_state == 1:
        print('Test passed successfully')
    elif test_state == -1:
        print('Test failed')
    else:
        print('No terminal state reached')
