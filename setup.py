#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools 

setuptools.setup(name='rlman',
                 version='0.0.1',
                 description='A software to implement RL method using baselines.',
                 author='Claudia D\'Ettorre',
                 author_email='c.dettorre@ucl.ac.uk',
                 license='MIT',
                 packages=['dVRL_simulator', 'dVRL_simulator.environments', 'dVRL_simulator.vrep'],
                 package_dir={
                     'dVRL_simulator' : 'dVRL_simulator',
                     'dVRL_simulator.envs' : 'dVRL_simulator/environments',
                     'dVRL_simulator.vrep' : 'dVRL_simulator/vrep',
                 },
                 install_requires=[
                     'gym==0.15.7',
                     'numpy',
                     'docker',
                     'matplotlib',
                     'transforms3d',
                     'scipy',
                     'pudb',
                ],
                include_package_data=True,
                zip_safe=False,
      )
