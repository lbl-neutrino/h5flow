#!/usr/bin/env python
import argparse
import yaml
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import h5flow.process

def main(input_filename, output_filename, config, start_position=None, end_position=None):
    with open(config,'r') as f:
        config = yaml.load(f)

    if rank == 0: print(config)
    manager = h5flow.process.H5FlowManager(input_filename, output_filename, config, start_position=start_position, end_position=end_position)

    if rank == 0: print('INIT')
    manager.init()

    if rank == 0: print('RUN')
    manager.run()

    if rank == 0: print('FINISH')
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
