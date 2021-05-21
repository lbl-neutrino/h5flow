#!/usr/bin/env python
import argparse
import yaml
import sys
import logging
from yaml import Loader
from mpi4py import MPI

from .core import H5FlowManager

def run(config, output_filename, input_filename=None, start_position=None, end_position=None, verbose=False):
    logging.basicConfig(format='%(asctime)s (p%(process)d) %(module)s.%(funcName)s[l%(lineno)d] %(levelname)s : %(message)s', level=logging.DEBUG if verbose else logging.WARNING)

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logging.info(f'output file: {output_filename}')
        logging.info(f'input file: {input_filename}')
        logging.info(f'start: {start_position}')
        logging.info(f'end: {end_position}')
        logging.info(f'verbose: {verbose}')
        logging.info('~~~ CONFIG DUMP ~~~')
        with open(config,'r') as f:
            for line in f.readlines():
                logging.info(line.strip('\n'))
        logging.info('~~~~~~~~~~~~~~~~~~~')
    with open(config,'r') as f:
        config = yaml.load(f, Loader=Loader)

    if rank == 0:
        logging.info('~~~ INIT ~~~')
    manager = H5FlowManager(config, output_filename, input_filename=input_filename, start_position=start_position, end_position=end_position)
    manager.init()
    if rank == 0:
        logging.info('~~~~~~~~~~~~')

    if rank == 0:
        logging.info('~~~ RUN ~~~')
    manager.run()
    if rank == 0:
        logging.info('~~~~~~~~~~~')

    if rank == 0:
        logging.info('~~~ FINISH ~~~')
    manager.finish()
    if rank == 0:
        logging.info('~~~~~~~~~~~~~~')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose','-v', action='store_true', help='''Increase verbosity''')
    parser.add_argument('--input_filename','-i', type=str, default=None, required=False, help='''input hdf5 file to loop over, optional if using a custom file generator''')
    parser.add_argument('--output_filename','-o', type=str, required=True)
    parser.add_argument('--config','-c', type=str, required=True, help='''yaml config file''')
    parser.add_argument('--start_position','-s', type=int, default=None, help='''start position within source dset (for partial file processing)''')
    parser.add_argument('--end_position','-e', type=int, default=None, help='''end position within source dset (for partial file processing)''')
    args = parser.parse_args()
    run(**vars(args))

