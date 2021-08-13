import pytest
import os

from h5flow import run, H5FLOW_MPI

if H5FLOW_MPI:
    @pytest.fixture
    def testfile(mpi_tmp_path):
        return os.path.join(mpi_tmp_path, 'test.h5')
else:
    @pytest.fixture
    def testfile(tmp_path):
        return os.path.join(tmp_path, 'test.h5')


def test_example(testfile):
    run('example_config.yaml', testfile, verbose=2)
