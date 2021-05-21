import logging

from h5flow.core import H5FlowStage

class H5FlowTestStage(H5FlowStage):
    '''
        A test stage that does nothing except load data from the ``requires``
        list and print arguments passed along with method.

        Example config::

            test:
                classname: H5FlowTestStage
                requires:
                    - 'example/dataset'

    '''
    def __init__(self, **params):
        logging.info('params:')
        for key,val in params.items():
            logging.info(f'\t{key}: {val}')

        super(H5FlowTestStage,self).__init__(**params)

    def init(self, source_name):
        logging.info(f'source_name: {source_name}')

        super(H5FlowTestStage,self).init(source_name)

    def run(self, source_name, source_slice, cache):
        logging.info(f'source_name: {source_name}')
        logging.info(f'source_slice: {source_slice}')
        logging.info('cache item lengths:')
        for key,val in cache.items():
            logging.info(f'\t{key}: {len(val)}')

        super(H5FlowTestStage,self).run(source_name, source_slice, cache)

