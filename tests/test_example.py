from h5flow import run


def test_example(testfile):
    run('example_config.yaml', testfile, verbose=2)
