import pytest
import os

from h5flow.data import H5FlowDataManager
from h5flow import H5FLOW_MPI

if H5FLOW_MPI:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    @pytest.fixture
    def testfile(mpi_tmp_path):
        path = os.path.join(mpi_tmp_path, 'test.h5')
        yield path
        if os.path.exists(path) and rank == 0:
            os.remove(path)
else:
    @pytest.fixture
    def testfile(tmp_path):
        path = os.path.join(tmp_path, 'test.h5')
        yield path
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def datamanager(testfile):
    dm = H5FlowDataManager(testfile)
    return dm
