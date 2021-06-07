import numpy as np
import numpy.ma as ma

ref_region_dtype = np.dtype([('start','i8'), ('stop','i8')])

def dereference(data, ref, region, sel=None, ref_direction=(0,1), as_masked=True):
    '''
        Load ``data`` referred to by ``ref`` that corresponds to the desired
        positions within the ``region`` lookup table.

        :param data: a ``h5py.Dataset`` or array to load dereferenced data from

        :param ref: a shape (N,2) ``h5py.Dataset`` or array of pairs of indices linking ``region`` and ``data``

        :param region: a 1D ``h5py.Dataset`` or array with a structured array type of [('start','i8'), ('stop','i8')]; 'start' defines the earliest index within the ``ref`` dataset, 'stop' defines the last index + 1 within the ``ref`` dataset

        :param sel: iterable of indices, an index, or a `slice` into a subset of the ``region`` dataset to load (optional)

        :param ref_direction: defines how to interpret second dimension of ``ref``. ``ref[:,ref_direction[0]]`` should produce indices into the ``region`` array, and ``ref[:,ref_direction[1]]`` should produce indices into the ``data`` array (``default=(0,1)``)

        :returns: ``numpy`` masked array, or (if ``as_masked=False``) ``list`` of length equivalent to ``sel``
    '''
    if sel is not None:
        region = region[sel] # load parent reference region information
    else:
        region = region[:]
        sel = slice(0,len(region))
    sel_idcs = np.r_[sel]

    ref_offset = np.min(region['start'])
    ref_sel = slice(ref_offset, int(np.max(region['stop'])))
    ref = ref[ref_sel] # load reference region

    dset_offset = np.min(ref[:,ref_direction[1]])
    dset_sel = slice(dset_offset, int(np.max(ref[:,ref_direction[1]])+1))
    dset = data[dset_sel] # load child dataset region

    if not as_masked:
        # dump into list using subregion masks
        data = [
                dset[ref[st:sp,ref_direction[1]][ (ref[st:sp,ref_direction[0]] == i) ] - dset_offset]
                for i,st,sp in zip(sel_idcs, region['start']-ref_offset, region['stop']-ref_offset)
            ]
        return data

    ref_mask = np.isin(ref[:,ref_direction[0]], sel_idcs) # only use relevant references

    uniq, counts = np.unique(ref[ref_mask,ref_direction[0]], return_counts=True) # get number of references per parent
    _,region_idcs,uniq_idcs = np.intersect1d(sel_idcs, uniq, assume_unique=True, return_indices=True) # map parent -> number of ref

    data = np.empty((len(sel_idcs), np.max(counts)), dtype=dset.dtype) # prep output array
    mask = np.zeros(data.shape, dtype=bool)

    mask[region_idcs] = np.arange(data.shape[-1]).reshape(1,-1) < counts[uniq_idcs].reshape(-1,1) # make n slots available per parent

    sort_ref = np.argsort(ref[ref_mask,ref_direction[0]]) # arrange by parent index
    np.place(data, mask=mask, vals=dset[ref[ref_mask,ref_direction[1]][sort_ref] - dset_offset]) # fill slots

    return ma.array(data, mask=~mask)

