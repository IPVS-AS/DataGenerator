import os
from setuptools import Distribution, find_packages, setup

with open('requirements.txt') as fh:
    requirements = [line.strip() for line in fh.readlines()]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup_kwargs = dict(
    name='DataGenerator',
    packages=find_packages(''),
    package_dir={'':'.'},
    version='0.1.0',
    install_requires=requirements,
    description='This repository contains the code for the submitted paper "Synthetic Data Generation for Imbalanced Multi-class Problems with Heterogeneous Groups".',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Dennis Treder-Tschechlov',
    author_email='dennis.tschechlov@ipvs.uni-stuttgart.de',
    license='MIT',
    python_requires=">=3.9",
)

setup(**setup_kwargs)