#!/usr/bin/env python3
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

setup(
    name='neuralretrieval',
    version='0.1.0',
    description='Neural Retrieval',
    long_description=readme,
    python_requires='>=3.6',
    packages=['data', 'models','utils'],
)
