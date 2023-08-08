from setuptools import setup, find_packages

setup(
    name="simpleSDM",
    version="1.0.0",
    author="Victor Simonsson",
    description="A short description of your project",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.4",
        "scipy>=1.10.1",
        "sounddevice>=0.4.6",
        "python-sofa>=0.2.0",
        "spaudiopy>=0.1.5",
        "requests>=2.31.0"

    ],
)