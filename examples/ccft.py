from beamngpy import BeamNGpy
from beamng_ncap.scenarios import CCFScenario, generate_scenario

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

    test = CCFScenario(beamng, 20, 55)
    sensors = test.load()
    test_state = test.execute('user')

    if test_state == 1:
        print('Test passed successfully')
    elif test_state == -1:
        print('Test failed')
    else:
        print('No terminal state reached')
