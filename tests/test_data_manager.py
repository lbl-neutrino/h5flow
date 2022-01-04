import pytest
import h5py
import os
import numpy as np

from h5flow.data import H5FlowDataManager
from h5flow.data import ref_region_dtype, dereference
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


def test_init(testfile, datamanager):
    # check that filepath was intialized correctly
    assert datamanager.filepath == testfile
    # check that file opens ok
    assert datamanager.fh
    # check that the file closes ok
    datamanager.close_file()
    assert not datamanager._fh
    # check that file (re)opens ok
    assert datamanager.fh


@pytest.fixture
def empty_testdset(datamanager):
    name = 'test/test'
    dm = datamanager
    dm.create_dset(name, int)
    return name


def test_create_dset(datamanager, empty_testdset):
    dm = datamanager
    # check that dataset opens and has correct type
    assert dm.get_dset(empty_testdset).dtype == int
    # check that dataset is still empty
    assert len(dm.get_dset(empty_testdset)) == 0


@pytest.fixture
def empty_testattr(datamanager, empty_testdset):
    dm = datamanager
    dm.set_attrs('test', test=123)
    return 'test'


def test_setattr(datamanager, empty_testattr):
    dm = datamanager
    # check that dataset was given correct attribute data
    assert dm.get_attrs(empty_testattr)['test'] == 123


@pytest.fixture
def empty_testref(datamanager, empty_testdset):
    dm = datamanager
    other_dset = empty_testdset + '_other'
    dm.create_dset(other_dset, dm.get_dset(empty_testdset).dtype)
    dm.create_ref(empty_testdset, other_dset)
    return empty_testdset, other_dset


def test_create_ref(datamanager, empty_testref):
    dm = datamanager
    # check that ref dataset opens and has correct type
    assert dm.get_ref(*empty_testref)[0].shape == (0, 2)
    assert dm.get_ref(*empty_testref)[-1] == (0, 1)
    assert dm.get_ref_region(*empty_testref).dtype == ref_region_dtype
    # check that ref dataset is still empty
    assert len(dm.get_ref(*empty_testref)[0]) == 0
    assert len(dm.get_ref_region(*empty_testref)) == 0


@pytest.fixture
def full_testdset(datamanager, empty_testdset, empty_testref):
    dm = datamanager
    sl = dm.reserve_data(empty_testdset, 100)
    dm.write_data(empty_testdset, sl, rank)
    return empty_testdset, sl


def test_write_dset(datamanager, full_testdset):
    dm = datamanager
    # check that we have access to the *full* dataset after writing
    assert len(dm.get_dset(full_testdset[0])) == size * 100
    # check that processes wrote to correct region
    assert all(dm.get_dset(full_testdset[0])[full_testdset[1]] == rank)


@pytest.fixture
def full_testref(datamanager, empty_testref, full_testdset):
    dm = datamanager
    sl = dm.reserve_data(empty_testref[-1], full_testdset[-1])
    dm.write_data(empty_testref[-1], sl, rank)
    ref_idcs = np.r_[full_testdset[1]].reshape(1, -1, 1)
    idcs = np.r_[full_testdset[1]].reshape(-1, 1, 1)
    idcs, ref_idcs = np.broadcast_arrays(idcs, ref_idcs)
    ref = np.concatenate((idcs, ref_idcs), axis=-1).reshape(-1, 2)
    dm.write_ref(*empty_testref, ref)
    return empty_testref, full_testdset[-1]


def test_write_ref(datamanager, full_testdset, full_testref):
    dm = datamanager
    n = len(dm.get_dset(full_testdset[0]))
    # check that we have access to the *full* ref dataset after writing
    assert len(dm.get_ref(*full_testref[0])[0]) == size * 100**2
    # check that child attribute is accessible and correct
    assert dm.fh[dm.get_ref(*full_testref[0])[0].attrs['dset0']] == dm.get_dset(full_testdset[0])
    assert dm.fh[dm.get_ref(*full_testref[0])[0].attrs['dset1']] == dm.get_dset(full_testref[0][-1])
    ref, ref_dir = dm.get_ref(*full_testref[0])
    # check that first of process' refs point to the correct chunk of the dataset
    sel = full_testref[1]
    ref_region = dm.get_ref_region(*full_testref[0])
    assert all(ref_region[sel]['start'] != ref_region[sel]['stop'])

    data = dereference(sel, ref, dm.get_dset(full_testdset[0]), ref_region, ref_direction=ref_dir)
    assert all([np.all(dm.get_dset(full_testdset[0])[sel] == d) for d in data])


def test_getitem(datamanager, full_testdset, full_testref):
    dm = datamanager
    assert dm.get_dset(full_testdset[0]) == dm[full_testdset[0] + '/data']
    assert len(dm[full_testdset[0], :10]) == 10
    assert np.all(dm[full_testdset[0], :10] == dm.get_dset(full_testdset[0])[:10])
    assert dm[tuple(full_testref[0])].dtype == dm.get_dset(full_testref[0][-1]).dtype
    assert len(dm[tuple(full_testref[0])]) == len(dm.get_dset(full_testref[0][0]))
    assert len(dm[full_testref[0][0], full_testref[0][1], :10]) == 10
    assert len(dm[full_testref[0][0], full_testref[0][1], 0]) == 1


def test_context(testfile):
    with H5FlowDataManager(testfile, 'a', mpi=H5FLOW_MPI) as dm:
        dm.create_dset('test', int)
        assert dm['test']


def test_repr(datamanager):
    print(datamanager)


@pytest.mark.skipif(size != 1, reason='test designed for single process only')
def test_write_ref(datamanager):
    dm = datamanager
    dm.create_dset('A', int)
    dm.create_dset('B', int)
    dm.create_ref('A', 'B')

    dm.reserve_data('A', 100)
    dm.reserve_data('B', 100)
    assert len(dm.get_dset('A')) == 100
    assert len(dm.get_dset('B')) == 100
    assert len(dm.get_ref_region('A', 'B')) == 100
    assert len(dm.get_ref_region('B', 'A')) == 100
    assert len(dm.get_ref('B', 'A')[0]) == 0

    # test writing A -> B references
    ref = np.c_[np.arange(0, 10), np.arange(20, 30)]
    dm.write_ref('A', 'B', ref)
    assert len(dm.get_dset('A')) == 100
    assert len(dm.get_dset('B')) == 100
    assert len(dm.get_ref_region('A', 'B')) == 100
    assert len(dm.get_ref_region('B', 'A')) == 100
    assert len(dm.get_ref('A', 'B')[0]) == 10
    assert len(dm.get_ref('B', 'A')[0]) == 10
    assert np.all(dm.get_ref('A', 'B')[0][:10, 0] == ref[:, 0])
    assert np.all(dm.get_ref('A', 'B')[0][:10, 1] == ref[:, 1])
    assert dm.get_ref('A', 'B')[1] == (0, 1)
    assert dm.get_ref('B', 'A')[1] == (1, 0)
    assert np.all(dm.get_ref_region('A', 'B')[0:10]['start'] == np.arange(10))
    assert np.all(dm.get_ref_region('A', 'B')[0:10]['stop'] == np.arange(10) + 1)
    assert np.all(dm.get_ref_region('B', 'A')[20:30]['start'] == np.arange(10))
    assert np.all(dm.get_ref_region('B', 'A')[20:30]['stop'] == np.arange(10) + 1)

    # test writing B -> A references
    ref = np.c_[np.arange(0, 10), np.arange(20, 30)]
    dm.write_ref('B', 'A', ref)
    assert len(dm.get_dset('A')) == 100
    assert len(dm.get_dset('B')) == 100
    assert len(dm.get_ref_region('A', 'B')) == 100
    assert len(dm.get_ref_region('B', 'A')) == 100
    assert len(dm.get_ref('A', 'B')[0]) == 20
    assert len(dm.get_ref('B', 'A')[0]) == 20
    assert np.all(dm.get_ref('B', 'A')[0][10:20, 1] == ref[:, 0])
    assert np.all(dm.get_ref('B', 'A')[0][10:20, 0] == ref[:, 1])
    assert dm.get_ref('A', 'B')[1] == (0, 1)
    assert dm.get_ref('B', 'A')[1] == (1, 0)
    assert np.all(dm.get_ref_region('A', 'B')[20:30]['start'] == np.arange(10, 20))
    assert np.all(dm.get_ref_region('A', 'B')[20:30]['stop'] == np.arange(10, 20) + 1)
    assert np.all(dm.get_ref_region('B', 'A')[0:10]['start'] == np.arange(10, 20))
    assert np.all(dm.get_ref_region('B', 'A')[0:10]['stop'] == np.arange(10, 20) + 1)
