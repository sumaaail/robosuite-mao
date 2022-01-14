import os


def ListAllSubdir(root_dir):
    for root, ds, fs in os.walk(root_dir):
        if fs != [] and ds == []:
            if 'Door' not in str.split(root, os.sep):
                for f in fs:
                    if f.endswith('.pkl'):
                        yield root


if __name__ == '__main__':
    root_dir = 'new_results'
    for i in ListAllSubdir(root_dir):
        print(i)
