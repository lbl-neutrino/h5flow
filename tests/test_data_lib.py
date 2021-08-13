import pytest
import os
import numpy as np
import numpy.ma as ma

from h5flow.data import H5FlowDataManager
from h5flow.data import *
from h5flow import H5FLOW_MPI

if H5FLOW_MPI:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    comm = None
    rank = 0
    size = 1


if H5FLOW_MPI:
    @pytest.fixture
    def testfile(mpi_tmp_path):
        return os.path.join(mpi_tmp_path, 'test.h5')
else:
    @pytest.fixture
    def testfile(tmp_path):
        return os.path.join(tmp_path, 'test.h5')


@pytest.fixture
def full_testfile(testfile):
    dm = H5FlowDataManager(testfile)
    dm.create_dset('A', int)
    dm.create_dset('B', int)
    dm.create_dset('C', int)
    dm.create_ref('A', 'B')
    dm.create_ref('C', 'B')

    data_a = np.arange(10 * rank, 10 * (rank + 1))
    data_b = 10 * np.broadcast_to(np.expand_dims(data_a, -1), (10, 10)) + np.expand_dims(np.arange(10), axis=0)
    data_c = 10 * np.broadcast_to(np.expand_dims(data_b, -1), (10, 10, 10)) + np.expand_dims(np.arange(10), axis=0)

    sl_a = dm.reserve_data('A', len(np.unique(data_a.ravel())))
    dm.write_data('A', sl_a, np.unique(data_a.ravel()))

    sl_b = dm.reserve_data('B', len(np.unique(data_b.ravel())))
    dm.write_data('B', sl_b, np.unique(data_b.ravel()))

    sl_c = dm.reserve_data('C', len(np.unique(data_c.ravel())))
    dm.write_data('C', sl_c, np.unique(data_c.ravel()))

    ref = np.unique(np.c_[np.broadcast_to(np.expand_dims(data_a, -1), data_b.shape).ravel(), data_b.ravel()], axis=0)
    dm.write_ref('A', 'B', ref)
    ref = np.unique(np.c_[data_c.ravel(), np.broadcast_to(np.expand_dims(data_b, -1), data_c.shape).ravel()], axis=0)
    dm.write_ref('C', 'B', ref)

    assert len(dm.fh['A/data']) == 10 * size
    assert len(dm.fh['B/data']) == 100 * size
    assert len(dm.fh['C/data']) == 1000 * size

    return dm.fh, ('A', 'B', 'C')


def test_print_ref(full_testfile):
    fh, _ = full_testfile
    print_ref(fh)


def test_print_data(full_testfile):
    fh, _ = full_testfile
    print_data(fh)


def test_dereference(full_testfile):
    fh, (a, b, c) = full_testfile

    sel = slice(0, 10)
    ref = fh[f'{a}/ref/{b}/ref']
    dset = fh[f'{b}/data']
    region = fh[f'{a}/ref/{b}/ref_region']

    data_no_reg = dereference(sel, ref, dset)

    data_reg = dereference(sel, ref, dset, region=region)

    data_idx = dereference(sel, ref, dset, region=region, indices_only=True)

    data_list = dereference(sel, ref, dset, region=region, as_masked=False)

    assert ma.all(data_no_reg == data_reg)
    assert data_reg.shape == (10, 10)
    assert data_reg.shape == data_idx.shape
    assert np.sum(data_reg.mask) == 0
    assert len(data_list) == 10
    assert isinstance(data_list, list)
    assert all([len(a) == 10 for a in data_list])


def test_dereference_chain(full_testfile):
    fh, (a, b, c) = full_testfile

    sel = slice(0, 10)
    refs = [fh[f'{a}/ref/{b}/ref'], fh[f'{c}/ref/{b}/ref']]
    dset = fh[f'{c}/data']
    regions = [fh[f'{a}/ref/{b}/ref_region'], fh[f'{b}/ref/{c}/ref_region']]
    ref_dir = [(0, 1), (1, 0)]

    data_no_reg = dereference_chain(sel, refs, dset)

    data_reg = dereference_chain(sel, refs, dset, regions=regions)

    data_idx = dereference_chain(sel, refs, dset, regions=regions, indices_only=True)

    assert ma.all(data_no_reg == data_reg)
    assert data_reg.shape == (10, 10, 1)
    assert data_reg.shape == data_idx.shape
    assert np.sum(data_reg.mask) == 0
