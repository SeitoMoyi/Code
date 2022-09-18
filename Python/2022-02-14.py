class Myexception():
    def __init__(self):
        import traceback
        import logging

        logging.basicConfig(
            filename = './error.log',
            format = '%(asctime)s %(levelname)s \n %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
        )

        logging.error(traceback.format_exc())

try:
    int('aa')
except:
    Myexception()