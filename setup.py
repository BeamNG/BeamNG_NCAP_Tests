from distutils.core import setup

setup(
    name='beamng_ncap',
    version='0.0.1',
    description='Recreation of Euro NCAP testing scenarios in BeamNG.drive',
    author='BeamNG GmbH',
    author_email='info@beamng.gmbh',
    packages=['beamng_ncap'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'beamngpy',
    ]
)
