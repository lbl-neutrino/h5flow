import h5py
import numpy as np
import numpy.ma as ma
import logging

ref_region_dtype = np.dtype([('start','i8'), ('stop','i8')])

def print_ref(grp):
    '''
        Print out all references in file (or group)
    '''
    l = list()
    grp.visititems(lambda n,d: l.append((n,d))\
        if n.endswith('/ref') and isinstance(d,h5py.Dataset)
        else None
        )
    max_length = max([len(n) for n,d in l])
    for n,d in l:
        print(n+' '*(max_length-len(n))+' '+str(d))

def print_data(grp):
    '''
        Print out all datasets in file (or group)

    '''
    l = list()
    grp.visititems(lambda n,d: l.append((n,d))\
        if n.endswith('/data') and isinstance(d,h5py.Dataset)
        else None
        )
    max_length = max([len(n) for n,d in l])
    for n,d in l:
        print(n+' '*(max_length-len(n))+' '+str(d))

def dereference(sel, ref, data, region=None, mask=None, ref_direction=(0,1), indices_only=False, as_masked=True):
    '''
        Load ``data`` referred to by ``ref`` that corresponds to the desired
        positions specified in ``sel``.

        :param sel: iterable of indices, an index, or a ``slice`` to match against ``ref[:,ref_direction[0]]``. Return value will have same first dimension as ``sel``, e.g. ``dereference(slice(100), ref, data).shape[0] == 100``

        :param ref: a shape (N,2) ``h5py.Dataset`` or array of pairs of indices linking ``sel`` and ``data``

        :param data: a ``h5py.Dataset`` or array to load dereferenced data from

        :param region: a 1D ``h5py.Dataset`` or array with a structured array type of [('start','i8'), ('stop','i8')]; 'start' defines the earliest index within the ``ref`` dataset for each value in ``sel``, and 'stop' defines the last index + 1 within the ``ref`` dataset (optional). If a ``h5py.Dataset`` is used, the ``sel`` spec will be used to load data from the dataset (i.e. ``region[sel]``), otherwise ``len(sel) == len(region)`` and a 1:1 correspondence is assumed

        :param mask: mask off specific items in selection (boolean, True == don't dereference selection), len(mask) == len(np.r_[sel])

        :param ref_direction: defines how to interpret second dimension of ``ref``. ``ref[:,ref_direction[0]]`` are matched against items in ``sel``, and ``ref[:,ref_direction[1]]`` are indices into the ``data`` array (``default=(0,1)``). So for a simple example: ``dereference([0,1,2], [[1,0], [2,1]], ['A','B','C','D'], ref_direction=(0,1))`` returns an array equivalent to ``[[],['A'],['B']]`` and ``dereference([0,1,2], [[1,0], [2,1]], ['A','B','C','D'], ref_direction=(1,0))`` returns an array equivalent to ``[['B'],['C'],[]]``

        :param indices_only: if ``True``, only returns the indices into ``data``, does not fetch data from ``data``

        :returns: ``numpy`` masked array (or if ``as_masked=False`` a ``list``) of length equivalent to ``sel``
    '''
    # set up selection
    sel_mask = mask
    sel_idcs = np.r_[sel][~sel_mask] if sel_mask is not None else np.r_[sel]
    n_elem = len(sel_idcs) if sel_mask is None else len(sel_mask)

    if not len(sel_idcs) and n_elem:
        # special case for if there is nothing selected in the mask
        if as_masked:
            return ma.array(np.empty((n_elem,1), dtype=data.dtype), mask=True)
        else:
            return [np.empty(0, data.dtype) for _ in range(n_elem)]
    elif not len(sel_idcs):
        if as_masked:
            return ma.array(np.empty((0,1), dtype=data.dtype), mask=True)
        else:
            return []

    # load fast region lookup
    if region is not None:
        if isinstance(region, h5py.Dataset):
            if isinstance(sel, slice):
                region = region[sel] # load parent reference region information
            else:
                region_offset = np.min(sel_idcs)
                region_sel = slice(region_offset, int(np.max(sel_idcs)+1))
                region = region[region_sel][sel_idcs - region_offset]
        else:
            region = region[sel_idcs]

    # load relevant references
    region_valid = region['start'] != region['stop'] if region is not None else None
    if not region is None and np.count_nonzero(region_valid) == 0:
        # special case for if there are no valid references
        if as_masked:
            return ma.array(np.empty((n_elem,1), dtype=data.dtype), mask=True)
        else:
            return [np.empty(0, data.dtype) for _ in range(n_elem)]
    ref_offset = np.min(region[region_valid]['start']) if region is not None else 0
    ref_sel = slice(ref_offset, int(np.max(region[region_valid]['stop']))) if region is not None else slice(ref_offset,len(ref))

    ref = ref[ref_sel] # load reference region

    dset_offset = np.min(ref[:,ref_direction[1]])
    dset_sel = slice(dset_offset, int(np.max(ref[:,ref_direction[1]])+1))
    dset = data[dset_sel] # load child dataset region

    if region is None:
        region = np.zeros(len(sel_idcs), dtype=ref_region_dtype)
        region['start'] = ref_sel.start
        region['stop'] = ref_sel.stop

    if not as_masked:
        # dump into list using subregion masks
        if indices_only:
            indices = [
                    ref[st:sp,ref_direction[1]][ (ref[st:sp,ref_direction[0]] == i) ]
                    for i,st,sp in zip(sel_idcs, region['start']-ref_offset, region['stop']-ref_offset)
                ]
            return indices
        else:
            data = [
                    dset[ref[st:sp,ref_direction[1]][ (ref[st:sp,ref_direction[0]] == i) ] - dset_offset]
                    for i,st,sp in zip(sel_idcs, region['start']-ref_offset, region['stop']-ref_offset)
                ]
            return data

    ref_mask = np.isin(ref[:,ref_direction[0]], sel_idcs) # only use relevant references

    uniq, counts = np.unique(ref[ref_mask,ref_direction[0]], return_counts=True) # get number of references per parent
    reordering = np.argsort(uniq) # sort references by parent index (used for filling the correct slots)
    uniq, counts = uniq[reordering], counts[reordering]
    max_counts = np.max(counts)

    # first fill a subarray consisting of unique elements that were requested (shape: (len(uniq_sel), max_counts) )
    uniq_sel, uniq_inv = np.unique(sel_idcs, return_inverse=True) # get mapping from unique reference parent -> selection index
    _,uniq_sel_idcs,uniq2uniq_sel_idcs = np.intersect1d(uniq_sel, uniq, assume_unique=False, return_indices=True) # map unique selection parent -> unique reference parent

    # set up arrays for unique selection parents
    shape = (len(uniq_sel), max_counts) + dset.shape[1:]
    condensed_data = np.zeros(shape, dtype=dset.dtype) if not indices_only \
        else np.zeros(shape, dtype=ref.dtype)
    condensed_mask = np.zeros(shape, dtype=bool)

    # block off and fill slots for unique selection
    condensed_mask[uniq_sel_idcs] = np.arange(condensed_data.shape[1]).reshape(1,-1) < counts[uniq2uniq_sel_idcs].reshape(-1,1)
    view_dtype = np.dtype([('ref0',ref.dtype),('ref1',ref.dtype)])
    sort_ref = np.argsort(ref[ref_mask].view(view_dtype), axis=0,
        order=[view_dtype.names[ref_direction[0]], view_dtype.names[ref_direction[1]]]
        ) # arrange by parent (then by child)
    # fill slots
    if indices_only:
        np.place(condensed_data, mask=condensed_mask, vals=ref[ref_mask,ref_direction[1]][sort_ref])
    else:
        np.place(condensed_data, mask=condensed_mask, vals=dset[ref[ref_mask,ref_direction[1]][sort_ref] - dset_offset])

    # then cast unique selections into full set of elements that were requested (shape: (len(sel), max_counts) )
    mask = np.zeros((len(sel_mask), max_counts), dtype=bool) if sel_mask is not None \
        else condensed_mask[uniq_inv]
    data = np.zeros((len(sel_mask), max_counts), dtype=condensed_data.dtype) if sel_mask is not None \
        else condensed_data[uniq_inv]
    if sel_mask is not None:
        mask[~sel_mask] = condensed_mask[uniq_inv]
        data[~sel_mask] = condensed_data[uniq_inv]
    return ma.array(data, mask=~mask)

