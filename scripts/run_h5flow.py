#!/usr/bin/env python
import argparse
import yaml
from yaml import Loader
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import h5flow.process

def main(input_filename, output_filename, config, start_position=None, end_position=None):
    with open(config,'r') as f:
        config = yaml.load(f, Loader=Loader)

    if rank == 0: print(config)
    manager = h5flow.process.H5FlowManager(input_filename, output_filename, config, start_position=start_position, end_position=end_position)

    print(f'\n~~~ INIT {rank}/{size} ~~~\n')
    manager.init()

    print(f'\n~~~ RUN {rank}/{size} ~~~\n')
    manager.run()

    print(f'\n~~~ FINISH {rank}/{size} ~~~\n')
    manager.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename','-i', type=str, required=True)
    parser.add_argument('--output_filename','-o', type=str, required=True)
    parser.add_argument('--config','-c', type=str, required=True, help='''yaml config file''')
    parser.add_argument('--start_position','-s', type=int, default=None, help='''start position within source dset (for partial file processing)''')
    parser.add_argument('--end_position','-e', type=int, default=None, help='''end position within source dset (for partial file processing)''')
    args = parser.parse_args()
    main(**vars(args))
