#!/usr/bin/env python
import logging
import os
import sys

global H5FLOW_MPI
if os.environ.get('H5FLOW_NOMPI', False):
    logging.warning(f'Running without mpi4py because H5FLOW_NOMPI={os.environ["H5FLOW_NOMPI"]}')
    H5FLOW_MPI = False
elif '--nompi' in sys.argv:
    logging.warning(f'Running without mpi4py because --nompi flag set')
    H5FLOW_MPI = False
else:
    try:
        from mpi4py import MPI
        H5FLOW_MPI = True
    except Exception as e:
        logging.warning(f'Running without mpi4py because {e}')
        H5FLOW_MPI = False

from .core import H5FlowManager, resources
import argparse
import yaml
from yamlinclude import YamlIncludeConstructor
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir='./')


def run(configs, output_filename, input_filename=None, start_position=None, end_position=None, verbose=0, drop=None, nompi=False):
    '''
        Execute a workflow specified by ``config`` writing to ``output_filename``.

        :param configs: ``list``, paths to configuration yamls to run in sequence

        :param output_filename: ``str``, path to output hdf5 file

        :param input_filename: ``str``, path to optional input file (default: ``None``)

        :param start_position: ``int``, loop start index given to generator (default: ``None``)

        :param end_position: ``int``, loop end index given to generator (default: ``None``)

        :param verbose: ``int``, verbosity level (``0 = warnings only``, ``1 = info``, ``2 = debug``)

        :param drop: ``list`` of ``str``, additional objects to drop from output file

        :param nompi: ``bool`` flag to force run without MPI

    '''
    global H5FLOW_MPI
    if nompi == True and H5FLOW_MPI:
        H5FLOW_MPI = False
    
    rank = MPI.COMM_WORLD.Get_rank() if H5FLOW_MPI else 0

    log_level = {0: 'WARNING', 1: 'INFO', 2: 'DEBUG'}[verbose]
    logging.basicConfig(format=f'%(asctime)s (r{rank}) %(module)s.%(funcName)s[l%(lineno)d] %(levelname)s : %(message)s', level=log_level)
    logging.getLogger().setLevel(log_level)

    if rank == 0:
        print('~~~ H5FLOW ~~~')
        print(f'output file: {output_filename}')
        if input_filename is not None:
            print(f'input file: {input_filename}')
        if start_position is not None:
            print(f'start: {start_position}')
        if end_position is not None:
            print(f'end: {end_position}')
        if drop is not None:
            print(f'drop: {drop}')
        if nompi is not False:
            print(f'no mpi: {nompi}')
        if verbose > 0:
            print(f'verbose: {log_level}')
        print('~~~~~~~~~~~~~~\n')

    for iconfig, config in enumerate(configs):
        # if running with multiple workflows, use output file for subsequent runs
        if iconfig != 0:
            input_filename = output_filename

        # refresh resource list
        global resources
        for key in list(resources.keys()):
            del resources[key]

        # load workflow configuration
        if rank == 0:
            print('~~~ WORKFLOW ~~~' if len(configs) == 1
                  else f'~~~ WORKFLOW ({iconfig+1}/{len(configs)}) ~~~')
            if verbose > 0:
                print(f'# {config}')
                with open(config, 'r') as f:
                    for line in f.readlines():
                        print(line.strip('\n'))
            else:
                print(config)
            print('~~~~~~~~~~~~~~~~\n')
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        if drop is not None:
            config['flow']['drop'] = config['flow'].get('drop', list()) + drop

        # execute workflow
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
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='''Increase verbosity, can specify more for more verbose (e.g. -vv)''')
    parser.add_argument('--input_filename', '-i', type=str, default=None,
                        required=False, help='''input hdf5 file to loop over, optional if using a custom file generator''')
    parser.add_argument('--output_filename', '-o', type=str, required=True)
    parser.add_argument('--configs', '-c', type=str, nargs='+', required=True,
                        help='''yaml config file(s)''')
    parser.add_argument('--start_position', '-s', type=int, default=None,
                        help='''start position within source dset (for partial file processing)''')
    parser.add_argument('--end_position', '-e', type=int, default=None,
                        help='''end position within source dset (for partial file processing)''')
    parser.add_argument('--drop', '-d', type=str, default=None, nargs='+',
                        help='''drop objects from output file''')
    parser.add_argument('--nompi', action='store_true', default=False,
                        help='''run h5flow without mpi enabled (can also be disabled by setting H5FLOW_NOMPI)''')
    args = parser.parse_args()

    run(**vars(args))
