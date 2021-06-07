import numpy as np
import numpy.ma as ma

ref_region_dtype = np.dtype([('start','i8'), ('stop','i8')])

def dereference(data, ref, region, sel=None, ref_direction=(0,1), as_masked=False):
    '''
        Load ``data`` referred to by ``ref`` that corresponds to the desired
        positions within the ``region`` lookup table.

        :param data: a ``h5py.Dataset`` or array to load dereferenced data from

        :param ref: a shape (N,2) ``h5py.Dataset`` or array of pairs of indices into first dimension of ``data``

        :param region: a 1D ``h5py.Dataset`` or array with a structured array type of [('start','i8'), ('stop','i8')]; 'start' defines the earliest index within the ``ref`` dataset, 'stop' defines the last index + 1 within the ``ref`` dataset

        :param sel: iterable of indices, an index, or a `slice` into a subset of the ``region`` dataset to load (optional)

        :param ref_direction: defines how indices in the second dimension of ``ref`` are treated. ``ref_direction[0]`` corresponds to the index in ``ref`` that points to
        ``ref[:,0]``, else load data corresponding to ``ref[:,1]``

        :returns: `list` of length equal to the spec, or ``numpy`` masked array (if ``as_masked==True``)
    '''
    if sel is not None:
        region = region[sel] # load reference region information
    else:
        region = region[:]
        sel = slice(0,len(region))

    ref_offset = np.min(region['start'])
    ref_sel = slice(ref_offset, int(np.max(region['stop'])))
    ref = ref[ref_sel] # load reference region

    dset_offset = np.min(ref[:,ref_direction[1]])
    dset_sel = slice(dset_offset, int(np.max(ref[:,ref_direction[1]])+1))
    dset = data[dset_sel] # load child dataset region

    data = [
            dset[ref[st:sp,ref_direction[1]][ (ref[st:sp,ref_direction[0]] == i) ] - dset_offset]
            for i,st,sp in zip(np.r_[sel], region['start']-ref_offset, region['stop']-ref_offset)
        ]
    if not as_masked:
        return data

    if len(data) == 0:
        return ma.empty((0,0), dtype=dset.dtype)
    max_ref = np.max([len(d) for d in data])
    masked_data = ma.empty((len(region),max_ref), dtype=dset.dtype)
    masked_data.mask = True
    for i,d in enumerate(data):
        n = len(d)
        masked_data[i,0:n] = d
        masked_data.mask[i,0:n] = False
    return masked_data
