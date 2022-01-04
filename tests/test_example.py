from h5flow import run


def test_example(testfile):
    run(['examples/example_config.yaml'], testfile, verbose=2)


def test_example_multi(testfile):
    run(['examples/example_config.yaml'] * 3, testfile, verbose=2)
