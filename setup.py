#!/usr/bin/env python

from distutils.core import setup

setup(name='h5flow',
      version='0.0',
      description='A workflow framework centered around hdf5 file access',
      author='Peter Madigan',
      author_email='pmadigan@berkeley.edu',
      packages=['h5flow'],
      scripts=['scripts/run_h5flow.py'],
      requires=[
        'numpy',
        'h5py',
        'mpi4py',
        'yaml',
        'tqdm',
        'pytest',
        'pytest-mpi'
      ]
     )
