import os


def list_dir(root_dir):
    found_files = []
    for path, dirs, files in os.walk(root_dir):
        for f in files:
            found_files.append(os.path.join(path, f))
    return found_files


def mkdir(dir):
    try:
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
    except OSError as e:
        print("[ERR] Creating directory error: {0}".format(e))
        exit(-1)


def mkdirs(*dirs):
    """
    :param dirs: a list of directories to create if these are not found
    :return exit_code: 0:success -1:failed
    """
    for dir in dirs:
        mkdir(dir)


def check_file(file_name):
    return os.path.isfile(file_name)


def isempty(dir):
    return len(os.listdir(dir)) == 0