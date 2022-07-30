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

            print('Preparing selection')
            for idset,dset in enumerate(datasets):
                # mark all entries that should be copied
                if entry_filter is not None:
                    filter_ = entry_filter[idset]

                    # default - select all
                    if filter_ is None:
                        entries = np.r_[slice(f_in.get_dset(dset).shape[0])]
                    
                    # check if specified by a list of entries
                    elif isinstance(filter_[0], int):
                        entries = filter_

                    # specified by a dataset within file
                    else:
                        filter_dset = f_in.get_dset(filter_[0])[:]
                        if filter_[1] is not None:
                            entries = np.where(filter_dset[filter_[1]].astype(bool))[0]
                        else:
                            entries = np.where(filter_dset.astype(bool))[0]
                else:
                    raise RuntimeError('No entry selection means specified')    

                to_process.append((dset, entries))
                print(f'Extract {len(entries)} entries from {dset}')

            pbar = tqdm.tqdm(total=len(to_process), position=0)
            while True:
                if not to_process:
                    break

                dset, entries = to_process[0]
                pbar.desc = dset + f' ({len(entries)}, {len(to_process)})'
                pbar.update()
                del to_process[0]
                # already finish processing dataset or empty entry list
                if dset in processed or not len(entries):
                    continue
                # last dataset chunk to process
                else:
                    processed.append(dset)

                if verbose:
                    print(f'Marking {dset} ({len(entries)})')
                flag_dset = dset + '~'
                # make boolean flag dataset (dset~)
                if flag_dset not in f_out['/']:
                    flag_dsets.append(flag_dset)
                    f_out.create_dset(flag_dset, dtype=bool)
                    f_out.reserve_data(flag_dset, f_in.get_dset(dset).shape[0])

                # mark entries that should be flagged
                for ibatch in tqdm.tqdm(range(len(entries)//batch_size + 1), position=1, leave=None, desc='Create flags'):
                    batch = entries[ibatch * batch_size:(ibatch + 1) * batch_size]
                    if not len(batch):
                        continue                    
                    f_out.write_data(flag_dset, batch, True)

                # get associated reference datasets
                refs = f_in.get_refs(dset)
                for ref in tqdm.tqdm(refs, leave=None, position=1, desc='Get referred entries'):
                    if verbose:
                        print(f'Marking reference {ref.name}')
                    # add referred data to processing queue
                    other_dset = ref.attrs['dset1'][1:-5] if ref.attrs['dset0'][1:-5] == dset else ref.attrs['dset0'][1:-5]
                    if other_dset in processed:
                        continue

                    child_entries = list()
                    for ibatch in tqdm.tqdm(range(len(entries)//batch_size + 1), position=2, leave=None, desc=f'{ref.name}'):
                        batch = entries[ibatch * batch_size:(ibatch + 1) * batch_size]
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
                        ref_entries = np.where(np.isin(ref[subset, ref_dir[0]], batch))[0]
                        if not len(ref_entries):
                            continue

                        # add children and entries to list of datasets to process
                        child = ref.attrs['dset0'][1:-5] if ref_dir[0] == 1 else ref.attrs['dset1'][1:-5]
                        offset = ref_entries.min()
                        child_batch_entries = np.unique(ref_dset[subset][ref_entries,ref_dir[1]]).astype(int)
                        child_entries.append(child_batch_entries)

                    if len(child_entries):
                        child_entries = np.sort(np.unique(np.concatenate(child_entries, axis=0)))
                        if len(child_entries):
                            if verbose:
                                print(f'Adding {child} with {len(child_entries)}')
                            to_process.append((child, child_entries))
                            pbar.total += 1
                            pbar.update(0)

            if verbose:
                print('Flag datasets:', flag_dsets)

            # copy marked data to output file
            def copy_data(path, obj):
                if isinstance(obj, h5py.Group):
                    if len(obj.attrs):
                        if verbose:
                            print(f'Copy attributes on {path}')
                        f_out.set_attrs(path, **dict(obj.attrs))
                    return

                name = path.removeprefix('/').removesuffix('/data').removesuffix('/ref').removesuffix('/ref_region')
                if path.endswith('/data'):
                    # make resizable dataset of correct type / shape
                    flag_dset_name = name + '~'
                    if flag_dset_name not in flag_dsets:
                        return
                    if verbose:
                        print(f'Copy data from {path}')
                    flag_dset = f_out.get_dset(flag_dset_name)
                    f_out.create_dset(name, dtype=obj.dtype, shape=obj.shape[1:])
                        
                    # copy entries
                    flag_index = np.where(flag_dset)[0]
                    for ibatch in tqdm.tqdm(range(len(flag_index)//batch_size + 1), desc=f'Copy {path}'):
                        batch = flag_index[ibatch * batch_size:(ibatch + 1) * batch_size]
                    
                        out_slice = f_out.reserve_data(name, len(batch))
                        offset = batch.min()
                        f_out.write_data(name, out_slice, obj[offset:batch.max()+1][batch-offset])

                    # copy attributes
                    if verbose:
                        print(f'Copy attributes on {path}')
                    f_out.set_attrs(path, **dict(obj.attrs))
                    return
                return

            def regen_ref(path, obj):
                if isinstance(obj, h5py.Group):
                    return
            
                name = path.removeprefix('/').removesuffix('/data').removesuffix('/ref').removesuffix('/ref_region')
                if path.endswith('/ref'):
                    # make resizable dataset of correct type / shape
                    parent_dset_name = obj.attrs['dset0'][1:-5]
                    child_dset_name = obj.attrs['dset1'][1:-5]

                    # grab appropriate index
                    parent_flag_dset_name = parent_dset_name + '~'
                    child_flag_dset_name = child_dset_name + '~'
                    if any([dset_name not in flag_dsets for dset_name in (parent_flag_dset_name, child_flag_dset_name)]):
                        return
                    if verbose:
                        print(f'Regenerating references from {path}')

                    # create reference dataset
                    f_out.create_ref(parent_dset_name, child_dset_name)
                        
                    parent_flag_dset = f_out.get_dset(parent_dset_name + '~')
                    child_flag_dset = f_out.get_dset(child_dset_name + '~')
                    parent_flag_index = np.where(parent_flag_dset)[0]
                    child_flag_index = np.where(child_flag_dset)[0]                    
                    parent_new_index = np.cumsum(parent_flag_dset)-1
                    child_new_index= np.cumsum(child_flag_dset)-1

                    # copy only references with both parent and child index
                    ref, ref_dir = f_in.get_ref(parent_dset_name, child_dset_name)
                    ref_region_parent = f_in.get_ref_region(parent_dset_name, child_dset_name)
                    for ibatch in tqdm.tqdm(range(len(parent_flag_index)//batch_size + 1), desc=f'Regenerate {path}'):
                        batch_indices = parent_flag_index[ibatch * batch_size:(ibatch + 1) * batch_size]
                        if not len(batch_indices):
                            continue

                        offset = batch_indices.min()
                        regions = ref_region_parent[offset:batch_indices.max()+1]
                        regions = regions[batch_indices - offset]
                        regions = regions[regions['start'] != regions['stop']]
                        if not len(regions):
                            continue

                        batch_ref = ref[regions['start'].min():regions['stop'].max()]
                        ref_sel = np.where(np.isin(batch_ref[:,ref_dir[0]], batch_indices) & np.isin(batch_ref[:,ref_dir[1]], child_flag_index))[0]
                        new_ref0 = parent_new_index[batch_ref[ref_sel,ref_dir[0]]]
                        new_ref1 = child_new_index[batch_ref[ref_sel,ref_dir[1]]]

                        new_ref = np.c_[new_ref0, new_ref1]
                        f_out.write_ref(parent_dset_name, child_dset_name, new_ref)

                    # copy attributes
                    f_out.set_attrs(path, **dict(obj.attrs))            
                    return
                return

            print('Copying data')
            f_in['/'].visititems(copy_data)
            print('Regenerating references')
            f_in['/'].visititems(regen_ref)

            print('Deleting flag datasets')
            for path in flag_dsets:
                if path in f_out['/']:
                    if verbose:
                        print(f'Delete {path}')
                    f_out.delete(path)

            print('Extracted data:')
            f_out['/'].visititems(lambda n,o: print(n, o.shape) if isinstance(o, h5py.Dataset) and (n.endswith('/data') or n.endswith('/ref') or n.endswith('/ref_region')) else None)

                
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
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='''Maximum entries to load at a given time''')
    # filter spec:
    # [
    #   (dataset, boolean field),
    # ]

    args = parser.parse_args()
    main(**vars(args))
