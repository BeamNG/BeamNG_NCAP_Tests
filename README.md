# BeamNG.tech NCAP Tests

## Introduction

This project implements some of the [Euro NCAP][1] testing scenarios for
automated vehicle systems in [BeamNG.tech][2] using the official
[BeamNG Python API][3]. Currently, only a handful of scenarios are implemented,
but more will be added in the future.

[1]: https://www.euroncap.com/en
[2]: https://beamng.tech/
[3]: https://github.com/BeamNG/BeamNGpy


## Installation 
- To install the project requirements, run the following command:
`pip install -r requirements.txt`
- To install the project as a package, run the following command:
`pip install -e .`

## Compatibility 

This version of the tool works with BeamNG.tech and BeamNGpy. Table of compatibility of different versions of BeamNG_NCAP_Tests is here:

| BeamNG_NCAP_Tests version                                    | BeamNG.tech version | BeamNGpy version                                          |
| ------------------------------------------------------------ | ------------------- | --------------------------------------------------------- |
| [0.2](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.2) | 0.35.5              | [1.32.0](https://github.com/BeamNG/BeamNGpy/tree/v1.32)   |
| [0.1](https://github.com/BeamNG/BeamNG_NCAP_Tests/tree/v0.1) | 0.34                | [1.31.0](https://github.com/BeamNG/BeamNGpy/tree/v1.31)   |

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

The NCAP scenarios are implemented to fit into user-defined vehicle setups. As
such you can define your own BeamNG.tech scenario to run the NCAP test in and
attach your own senor models to the vehicle being tested.
