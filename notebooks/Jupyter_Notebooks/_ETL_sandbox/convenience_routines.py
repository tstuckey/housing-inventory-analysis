from urllib.request import urlretrieve
import hashlib as hl
import sys


def get_remote_db_file(t_url: str, t_local_file: str) -> None:
    """
    Get the binary database file from the remote and write it to the local file system
    :param t_url:
    :param t_local_file:
    :return:
    """
    pass
    return


def compute_md5_hash(t_path: str) -> str:
    """
    Compute the MD5 hash of the db file passed in
    :param t_path: db file path
    :return:
    """
    return hl.md5(open(t_path, 'rb').read()).hexdigest()


def get_and_check_db(t_url: str, t_md5_str) -> None:
    # TODO check if the database is already on the local file system
    #   if it's not already there, download it from the URL
    # TODO check the hash on the database file against what we're supposed
    #  to have, if it's wrong, download from the URL again
    pass
    return


if __name__ == "__main__":
    print(compute_md5_hash(sys.argv[1]))