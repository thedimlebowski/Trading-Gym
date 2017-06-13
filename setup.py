#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='openoutcry',
    version='0.1.0',
    description="Open Outcry is an open-source project for the development of reinforcement learning algorithms in the context of trading.",
    author="Prediction Machines",
    author_email='openoutcry@prediction-machines.com',
    url='https://github.com/prediction-machines/openoutcry',
    packages=find_packages(),
    install_requires=[
        'matplotlib==2.0.2'
    ],
    license="MIT license",
    zip_safe=False,
    keywords='openoutcry'
)
