# BeamNG.tech NCAP Tests

## Introduction

This project implements some of the [Euro NCAP][1] testing scenarios for
automated vehicle systems in [BeamNG.tech][2] using the official
[BeamNG Python API][3]. Currently, only a handful of scenarios are implemented,
but more will be added in the future.Currently, several Car-to-Car Rear (CCR) scenarios from the official Euro NCAP 
AEB test suite are implemented, with more scenarios planned for future releases.

## File Structure

```
├── src/
│   ├── AEB_test_template.ipynb    # Main test notebook
│   └── beamng_ncap/
│       ├── __init__.py
│       ├── scenarios.py           # Core scenario implementations
│       └── controllers.py         # PID and trial controllers
├── examples/
│   ├── ccrs.py                    # CCRs test example
│   ├── ccrm.py                    # CCRm test example
│   ├── ccrb.py                    # CCRb test example
│   └── ccftap.py                  # CCFTAP test 
|
├── setup.py                       # Package 
├── requirements.txt               # Dependencies
├── README.md                      
```


## Implemented Scenarios

The project currently includes the following standardized Euro NCAP ADAS test scenarios:

### Car-to-Car Rear (CCR) Scenarios

- **CCRs** (`examples/ccrs.py`) - **Car-to-Car Rear Stationary**  
  A collision in which a vehicle travels forwards towards another stationary vehicle and the frontal structure of the vehicle strikes the rear structure of the other.

- **CCRm** (`examples/ccrm.py`) - **Car-to-Car Rear Moving**  
  A collision in which a vehicle travels forwards towards another vehicle that is travelling at constant speed and the frontal structure of the vehicle strikes the rear structure of the other.

- **CCRb** (`examples/ccrb.py`) - **Car-to-Car Rear Braking**  
  A collision in which a vehicle travels forwards towards another vehicle that is travelling at constant speed and then decelerates, and the frontal structure of the vehicle strikes the rear structure of the other.

### Car-to-Car Front (CCF) Scenarios

- **CCFtap** (`examples/ccftap.py`) - **Car-to-Car Front Turn-Across-Path**  
  A collision in which a vehicle turns across the path of an oncoming vehicle travelling at constant speed, and the frontal structure of the vehicle strikes the front structure of the other.

## Installation 
- To install the project requirements, run the following command:
`pip install -r requirements.txt`
- To install the project as a package, run the following command:
`pip install -e .`

## Compatibility 

This version of the tool works with BeamNG.tech and BeamNGpy. Table of compatibility of different versions of BeamNG_NCAP_Tests is here:

| BeamNG_NCAP_Tests version                                    | BeamNG.tech version   | BeamNGpy version                                          |
| ------------------------------------------------------------ | --------------------- | --------------------------------------------------------- |
| [0.3.0](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.3.0) | 0.36                | [1.33.0](https://github.com/BeamNG/BeamNGpy/tree/v1.33)   |
| [0.2.1](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.2.1) | 0.36                | [1.34.0](https://github.com/BeamNG/BeamNGpy/tree/v1.34)   |
| [0.2.0](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.2) | 0.35.5              | [1.32.0](https://github.com/BeamNG/BeamNGpy/tree/v1.32)   |
| [0.1.0](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.1) | 0.34                | [1.31.0](https://github.com/BeamNG/BeamNGpy/tree/v1.31)   |

Other versions of BeamNG.tech and BeamNGpy will not work with this version.

## Usage Example

After installing the library, the scenarios become available in
the `beamng_ncap.scenarios` module. They can be executed as such:

```python
from beamngpy import BeamNGpy
from beamng_ncap.scenarios import CCRS, generate_scenario

beamng = BeamNGpy('localhost', 25252, home='/path/to/bng/tech', user='/path/to/bng/tech/userfolder')
with beamng as bng:
    scenario = generate_scenario('etk800', 'etk800')
    scenario.make(bng)
    bng.set_deterministic()
    bng.set_steps_per_second(100)
    bng.load_scenario(scenario)
    bng.pause()
    bng.start_scenario()

    overlap = 100  # -75, -50, 50, 75 or 100 %
    vut_speed = 45  # 10, 15, 20, 25, 30, 35, 40, 45 or 50 km/h
    test = CCRS(bng, vut_speed, overlap)
    # Add custom sensors here (electrics, damage & timer are already attached)
    # test.vut.attach_sensor(...)
    sensors = test.load()
    input('Press enter when done...')
```

The NCAP scenarios are implemented to fit into user-defined vehicle setups. As such you can define your own BeamNG.tech scenario to run the NCAP test in and attach your own sensor models to the vehicle being tested.

## References

For detailed information about Euro NCAP AEB testing protocols, refer to:
- [Euro NCAP Official Website](https://www.euroncap.com/en)
- [Euro NCAP AEB Car-to-Car Test Protocol v4.0](https://cdn.euroncap.com/media/67887/euro-ncap-aeb-c2c-test-protocol-v40.pdf) - Official test protocol document (February 2022) defining the standardized test scenarios for Autonomous Emergency Braking systems



[1]: https://www.euroncap.com/en
[2]: https://beamng.tech/
[3]: https://github.com/BeamNG/BeamNGpy