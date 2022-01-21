import os
import sys


def ListAllSubdir(root_dir):
    for root, ds, fs in os.walk(root_dir):
        if fs != [] and ds == []:
            if 'Door' not in str.split(root, os.sep):
                for f in fs:
                    if f.endswith('.json'):
                        yield root


if __name__ == '__main__':
    print(sys._getframe().f_code.co_filename)
    print(sys._getframe(0).f_code.co_filename)
    print(sys._getframe().f_lineno)
    print(sys._getframe(0).f_lineno)
