from setuptools import setup

setup(
    name="gym_dimensional_doors",
    version="0.1.0",
    url="https://github.com/DBALICKI/gym-dimensional-doors",
    author="Daniel Balicki",
    license="MIT",
    packages=["gym_dimensional_doors", "gym_dimensional_doors.envs"],
    install_requires=["gym", "numpy"],
)
