#!/usr/bin/env python
import logging
try:
    from mpi4py import MPI
    H5FLOW_MPI = True
except Exception as e:
    logging.warning(f'Running without mpi4py because {e}')
    H5FLOW_MPI = False
from .core import H5FlowManager, resources
import argparse
import yaml
import sys
from yamlinclude import YamlIncludeConstructor
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='./')


def run(config, output_filename, input_filename=None, start_position=None, end_position=None, verbose=0):
    '''
        Execute a workflow specified by ``config`` writing to ``output_filename``.

        :param config: ``str``, path to configuration yaml

        :param output_filename: ``str``, path to output hdf5 file

        :param input_filename: ``str``, path to optional input file (default: ``None``)

        :param start_position: ``int``, loop start index given to generator (default: ``None``)

        :param end_position: ``int``, loop end index given to generator (default: ``None``)

        :param verbose: ``int``, verbosity level (``0 = warnings only``, ``1 = info``, ``2 = debug``)

    '''
    rank = MPI.COMM_WORLD.Get_rank() if H5FLOW_MPI else 0

    log_level = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}[verbose]
    logging.basicConfig(format=f'%(asctime)s (r{rank}) %(module)s.%(funcName)s[l%(lineno)d] %(levelname)s : %(message)s', level=log_level)

    global resources
    # refresh resource list
    for key in list(resources.keys()):
        del resources[key]

    if rank == 0:
        print('~~~ H5FLOW ~~~')
        print(f'output file: {output_filename}')
        print(f'input file: {input_filename}')
        print(f'start: {start_position}')
        print(f'end: {end_position}')
        print(f'verbose: {verbose}')
        print('~~~~~~~~~~~~~~\n')
        print('~~~ WORKFLOW ~~~')
        with open(config, 'r') as f:
            for line in f.readlines():
                print(line.strip('\n'))
        print('~~~~~~~~~~~~~~~~\n')
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if rank == 0:
        print('~~~ INIT ~~~')
    manager = H5FlowManager(config, output_filename, input_filename=input_filename, start_position=start_position, end_position=end_position)
    manager.init()
    if rank == 0:
        print('~~~~~~~~~~~~\n')

    if rank == 0:
        print('~~~ RUN ~~~')
    manager.run()
    if rank == 0:
        print('~~~~~~~~~~~\n')

    if rank == 0:
        print('~~~ FINISH ~~~')
    manager.finish()
    if rank == 0:
        print('~~~~~~~~~~~~~~\n')


def main():
    '''
        Entry point for command line execution. Parses arguments from command
        line and executes ``run(**parsed_args)``.

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0, help='''Increase verbosity, can specify more for more verbose (e.g. -vv)''')
    parser.add_argument('--input_filename', '-i', type=str, default=None, required=False, help='''input hdf5 file to loop over, optional if using a custom file generator''')
    parser.add_argument('--output_filename', '-o', type=str, required=True)
    parser.add_argument('--config', '-c', type=str, required=True, help='''yaml config file''')
    parser.add_argument('--start_position', '-s', type=int, default=None, help='''start position within source dset (for partial file processing)''')
    parser.add_argument('--end_position', '-e', type=int, default=None, help='''end position within source dset (for partial file processing)''')
    args = parser.parse_args()

    run(**vars(args))
