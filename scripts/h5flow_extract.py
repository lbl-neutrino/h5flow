import h5py
import numpy as np
import argparse
import json
import tqdm

import h5flow
from h5flow.data import H5FlowDataManager


def main(**kwargs):
    verbose = kwargs['verbose']
    input_filename = kwargs['input_filename']
    output_filename = kwargs['output_filename']
    datasets = kwargs['datasets']
    batch_size = kwargs['batch_size']    
    entry_filter = kwargs['filter']
    if entry_filter is None:
        entry_filter = [None] * len(datasets)

    to_process = []
    processed = []
    flag_dsets = []    

    # open the input file
    with H5FlowDataManager(input_filename, 'r') as f_in:
        # create a copied file
        with H5FlowDataManager(output_filename, 'a') as f_out:

            print('Preparing selection') if f_out.rank == 0 else None
            for idset,dset in enumerate(datasets):
                # mark all entries that should be copied
                if entry_filter is not None:
                    filter_ = entry_filter[idset]

                    # default - select all
                    if filter_ is None:
                        entries = np.r_[slice(f_in.get_dset(dset).shape[0])].astype(int)
                    
                    # check if specified by a list of entries
                    elif isinstance(filter_[0], int):
                        entries = np.array(filter_, dtype=int)

                    # specified by a dataset within file
                    else:
                        filter_dset = f_in.get_dset(filter_[0])[:]
                        if filter_[1] is not None:
                            entries = np.where(filter_dset[filter_[1]].astype(bool))[0].astype(int)
                        else:
                            entries = np.where(filter_dset.astype(bool))[0].astype(int).tolist()
                else:
                    raise RuntimeError('No entry selection means specified')    

                to_process.append((dset, entries))
                print(f'Extract {len(entries)} entries from {dset}') if f_out.rank == 0 else None

            pbar = tqdm.tqdm(total=len(to_process), position=0, disable=f_out.rank != 0)
            while True:
                if not to_process:
                    break

                dset, entries = to_process[0]
                pbar.desc = dset + f' {len(entries)} elem, {len(to_process)} dsets remaining'
                pbar.update()
                del to_process[0]
                # already finish processing dataset or empty entry list
                if dset in processed or not len(entries):
                    continue
                # last dataset chunk to process
                else:
                    processed.append(dset)

                flag_dset = dset + '~'
                # make boolean flag dataset (dset~)
                if flag_dset not in flag_dsets:
                    flag_dsets.append(flag_dset)
                    f_out.create_dset(flag_dset, dtype=bool)
                    f_out.reserve_data(flag_dset, f_in.get_dset(dset).shape[0])

                # mark entries that should be flagged
                batch_size_ = max(batch_size // entries.dtype.itemsize, 1)
                for ibatch in tqdm.tqdm(range(f_out.rank, len(entries)//batch_size_ + 1 + f_out.rank,  f_out.size), position=1, leave=None, desc=f'Create flags {batch_size_}', disable=f_out.rank != 0):
                    batch = entries[ibatch * batch_size_:(ibatch + 1) * batch_size_]
                    if not len(batch):
                        batch = slice(-1,-1)
                    f_out.write_data(flag_dset, batch, True)

                # get associated reference datasets
                refs = f_in.get_refs(dset)
                for ref in tqdm.tqdm(refs, leave=None, position=2, desc='Collect referred dsets', disable=f_out.rank != 0):
                    # add referred data to processing queue
                    other_dset = ref.attrs['dset1'][1:-5] if ref.attrs['dset0'][1:-5] == dset else ref.attrs['dset0'][1:-5]
                    
                    if other_dset in processed:
                        continue

                    child_entries = [list()]
                    ref_region = f_in.get_ref_region(dset, other_dset)
                    batch_size_ = max(batch_size // int(np.max((ref_region[:batch_size]['stop'] - ref_region[:batch_size]['start']) * 2 * ref.dtype.itemsize).clip(1,None)), 1)
                    for ibatch in tqdm.tqdm(range(f_out.rank, len(entries)//batch_size_ + f_out.rank + 1, f_out.size), position=3, leave=None, desc=f'{ref.name} {batch_size_}', disable=f_out.rank != 0):
                        batch = entries[ibatch * batch_size_:(ibatch + 1) * batch_size_]

                        if not len(batch):
                            continue

                        offset = batch.min()
                        ref_region = f_in.get_ref_region(dset, other_dset)[offset:batch.max()+1]
                        ref_region = ref_region[batch - offset]
                        ref_region = ref_region[ref_region['start'] != ref_region['stop']]

                        if not len(ref_region):
                            continue
                        subset = slice(ref_region['start'].min(), ref_region['stop'].max())
                        
                        # mark data referred by entries
                        ref_dset,ref_dir = f_in.get_ref(dset, other_dset)
                        ref_entries = np.where(np.isin(ref[subset, ref_dir[0]], batch))[0].astype(int)
                        if not len(ref_entries):
                            continue
                        # add children and entries to list of datasets to process
                        offset = ref_entries.min()
                        child_batch_entries = np.unique(ref_dset[subset][ref_entries,ref_dir[1]]).astype(int)
                        child_entries.append(child_batch_entries)
                    if h5flow.H5FLOW_MPI:
                        child_entries = f_out.comm.allgather(child_entries)
                        child_entries = [v for vs in child_entries for v in vs]

                    if len(child_entries):
                        child_entries = np.sort(np.unique(np.concatenate(child_entries, axis=0))).astype(int)
                        if len(child_entries):
                            if verbose and f_out.rank == 0:
                                print(f'Adding {other_dset} with {len(child_entries)}')
                            to_process.append((other_dset, child_entries))
                            pbar.total += 1
                            pbar.update(0)
            if f_out.rank == 0:
                print()
            if verbose and f_out.rank == 0:
                print('Flag datasets:', flag_dsets)

            # copy marked data to output file
            def copy_data(path, obj):
                if isinstance(obj, h5py.Group):
                    if len(obj.attrs):
                        if verbose and f_out.rank == 0:
                            print(f'Copy attributes on {path}')
                        f_out.set_attrs(path, **dict(obj.attrs))
                    return

                if path.endswith('/data'):
                    # make resizable dataset of correct type / shape
                    name = path[:-5]
                    flag_dset_name = name + '~'
                    if flag_dset_name not in flag_dsets:
                        return
                    if verbose and f_out.rank == 0:
                        print(f'Copy data from {path}')
                    flag_dset = f_out.get_dset(flag_dset_name)
                    f_out.create_dset(name, dtype=obj.dtype, shape=obj.shape[1:])
                        
                    # copy entries
                    batch_size_ = max(batch_size // obj.dtype.itemsize, 1)
                    flag_batch_size_ = max(batch_size // flag_dset.dtype.itemsize, 1)
                    pbar = tqdm.tqdm(total=len(flag_dset), desc=f'Copy {path} {batch_size_}', disable=f_out.rank != 0)
                    for ibatch_flag in range(len(flag_dset)//flag_batch_size_ + 1):
                        flag_batch = flag_dset[ibatch_flag * flag_batch_size_:(ibatch_flag+1) * flag_batch_size_]
                        flag_index = np.where(flag_batch)[0].astype(int) + ibatch_flag * flag_batch_size_

                        for ibatch in range(f_out.rank, len(flag_index)//batch_size_ + f_out.rank + 1, f_out.size):
                            batch = flag_index[ibatch * batch_size_:(ibatch + 1) * batch_size_]
                            out_slice = f_out.reserve_data(name, len(batch))
                            copy_data = np.empty((0,)+obj.shape[1:], dtype=obj.dtype)
                            if len(batch):
                                offset = batch.min()
                                copy_data = obj[offset:batch.max()+1][batch-offset]
                            f_out.write_data(name, out_slice, copy_data)
                        pbar.update(len(flag_batch))

                    # copy attributes
                    if verbose and f_out.rank == 0:
                        print(f'Copy attributes on {path}')
                    f_out.set_attrs(path, **dict(obj.attrs))
                    return
                return

            def calc_new_index(idx, flag_dset, cache=None):
                # ``idx`` is a 1d index array marking the indices we want to convert
                # ``flag_dset`` is a boolean dataset marking entries that should be counted towards new indexing
                # ``cache`` is a stored old index, new index mapping
                #
                # returns an array of the same shape as ``idx``, but with values representing their position within a masked array: ``arr[flag_dset][return_idx] == arr[idx]``
                if cache is None:
                    cache = [(0,0)]
                if len(idx):
                    min_idx = idx.min()

                    cache_entry = cache[-1]
                    while min_idx < cache_entry[0]:
                        if cache:
                            del cache[-1]
                        if cache:
                            cache_entry = cache[-1]
                        else:
                            cache_entry = (0,0)
                            break
                            
                    offset = min(min_idx, cache_entry[0])
                    batch = flag_dset[offset:idx.max()+1].astype(int)
                    new_idx = (np.cumsum(batch)-1 + cache_entry[1])[idx-offset]

                    cache.append((idx.max(), new_idx.max()))
                    return new_idx, cache
                return np.zeros_like(idx), cache
                        
            def regen_ref(path, obj):
                if isinstance(obj, h5py.Group):
                    return
            
                if path.endswith('/ref'):
                    # make resizable dataset of correct type / shape
                    parent_dset_name = obj.attrs['dset0'][1:-5]
                    child_dset_name = obj.attrs['dset1'][1:-5]

                    # grab appropriate index
                    parent_flag_dset_name = parent_dset_name + '~'
                    child_flag_dset_name = child_dset_name + '~'
                    if any([dset_name not in flag_dsets for dset_name in (parent_flag_dset_name, child_flag_dset_name)]):
                        return
                    if verbose and f_out.rank == 0:
                        print(f'Regenerating references from {path}')

                    # create reference dataset
                    f_out.create_ref(parent_dset_name, child_dset_name)
                        
                    parent_flag_dset = f_out.get_dset(parent_dset_name + '~')
                    child_flag_dset = f_out.get_dset(child_dset_name + '~')

                    # copy only references with both parent and child index
                    ref, ref_dir = f_in.get_ref(parent_dset_name, child_dset_name)

                    batch_size_ = max(batch_size // int(max(2 * obj.dtype.itemsize * f_out.size, 1)), 1)
                    parent_cache = None
                    child_cache = None
                    for ibatch in tqdm.tqdm(range(f_out.rank, len(ref)//batch_size_ + f_out.rank + 1, f_out.size), desc=f'Regen {path} {batch_size_}', disable=f_out.rank != 0):
                        batch_ref = ref[ibatch*batch_size_:(ibatch+1)*batch_size_]
                        new_ref = batch_ref.copy().astype(int)
                        if len(new_ref):
                            offset = new_ref[:,ref_dir[0]].min()
                            end = new_ref[:,ref_dir[0]].max()+1
                            batch_parent_flag = parent_flag_dset[offset:end][new_ref[:,ref_dir[0]]-offset].astype(bool)
                            new_ref = new_ref[batch_parent_flag]
                            if len(new_ref):
                                offset = new_ref[:,ref_dir[1]].min()
                                end = new_ref[:,ref_dir[1]].max()+1
                                batch_child_flag = child_flag_dset[offset:end][new_ref[:,ref_dir[1]]-offset].astype(bool)
                                new_ref = new_ref[batch_child_flag]
                                if len(new_ref):
                                    new_ref[:,ref_dir[0]], parent_cache = calc_new_index(new_ref[:,ref_dir[0]], parent_flag_dset, cache=parent_cache)
                                    new_ref[:,ref_dir[1]], child_cache = calc_new_index(new_ref[:,ref_dir[1]], child_flag_dset, cache=child_cache)
                        f_out.write_ref(parent_dset_name, child_dset_name, new_ref)

                    # copy attributes
                    f_out.set_attrs(path, **dict(obj.attrs))
                    return
                return

            print('Copying data') if f_out.rank == 0 else None
            f_in['/'].visititems(copy_data)
            print('Regenerating references') if f_out.rank == 0 else None
            f_in['/'].visititems(regen_ref)

            print('Deleting flag datasets') if f_out.rank == 0 else None
            for path in flag_dsets:
                if path in f_out['/']:
                    if verbose and f_out.rank == 0:
                        print(f'Delete {path}')
                    f_out.delete(path)

            if f_out.rank == 0:
                print('Extracted data:')
                f_out['/'].visititems(lambda n,o: print(n, o.shape) if isinstance(o, h5py.Dataset) and (n[-5:] == '/data' or n[-4:] == '/ref' or n[-11:] == '/ref_region') else None)

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='''Increase verbosity of script''')
    parser.add_argument('--input_filename', '-i', type=str, required=True,
                        help='''Input hdf5 file path to extract data from''')
    parser.add_argument('--output_filename', '-o', type=str, required=True,
                        help='''Output hdf5 file to store extracted data''')
    parser.add_argument('--datasets', '-d', type=str, nargs='+', required=True,
                        help='''Datasets to copy''')
    parser.add_argument('--filter', '-f', type=json.loads, default=None,
                        help='''JSON-formatted filter spec for each dataset''')
    parser.add_argument('--batch_size', type=int, default=1024*1024,
                        help='''Maximum data to handle at a given time (bytes)''')
    # filter spec:
    # [
    #   (dataset, boolean field),
    # ]

    args = parser.parse_args()
    main(**vars(args))
