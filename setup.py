import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('VERSION', 'r') as fh:
    version = fh.read().strip()

setuptools.setup(name='h5flow',
    version=version,
    description='A workflow framework centered around hdf5 file access',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Peter Madigan',
    author_email='pmadigan@berkeley.edu',
    package_dir={'': '.'},
    packages=setuptools.find_packages(where='.'),
    python_requires='>=3.7',
    entry_points={'console_scripts': ['h5flow=h5flow:main']},
    scripts=['scripts/run_h5flow.py'],
    install_requires=[
      'numpy',
      'h5py~=2.10',
      'mpi4py',
      'PyYAML',
      'tqdm',
      'pytest',
      'pytest-mpi',
    ]
   )
