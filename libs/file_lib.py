import os
def os_path(path):
    path_elements = path.split('/')
    return os.path.join(*path_elements)