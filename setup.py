import setuptools
from setuptools import setup, find_packages
import os


# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="commonroad_repairer",
    version="0.0.0.dev4",
    author="Cyber-Physical Systems Group, Technical University of Munich",
    author_email="commonroad@lists.lrz.de",
    description="It's pip... with git.",
    long_description=long_description,
    include_package_data=True,
    url="https://gitlab.lrz.de/yuanfei/commonroad_repair",
    packages=setuptools.find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD License",
        "Operating System :: OS Independent",
    ],
)
