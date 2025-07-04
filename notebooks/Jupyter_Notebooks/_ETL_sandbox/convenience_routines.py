from urllib.request import urlretrieve
import pathlib as pl
import hashlib as hl
import sqlite3 as sq
import os

db_path = '../../_data/'
db_url = 'https://storage.googleapis.com/housing-inventory-storage.simplifyingcomplexspaces.com/'
db_name = 'housing_inventory.db'
md5_val = '983701daac8846f3fcf3f93d73d2f1f9'


def compute_md5_hash(t_path: str) -> str:
    """
    Compute the MD5 hash of the db file passed in
    :param t_path: db file path
    :return:
    """
    return hl.md5(open(t_path, 'rb').read()).hexdigest()


def get_and_check_db_file(local_filename=db_path + db_name,
                          full_url=db_url + db_name,
                          hash_val=md5_val) -> None:
    """
    Check to see if we have the database file locally. If we don't have it, download it.
    Check the file hash is correct. If it's not right, try downloading it; then check the hash again
    :return:
    """
    print('\nFrom convenience routine, Working Directory is:\t' + str(pl.Path.cwd()))
    print('\nFrom convenience routine,  local file name is:\t' + local_filename + ' and pl.Path version is:\t' +
          str(pl.Path(local_filename)) + '\n')

    if not pl.Path(local_filename).exists():
        urlretrieve(full_url, local_filename)
    else:
        print('already had a copy')

    # check the hash of the local copy regardless whether we had it or we just downloaded it
    file_hash = compute_md5_hash(local_filename)
    if not file_hash == hash_val:
        # hashes didn't match, let's try to download it again
        urlretrieve(full_url, local_filename)
        file_hash = compute_md5_hash(local_filename)
        if file_hash != hash_val:
            print('hash mismatch even with a fresh pull, double check the hash in your notebook')
            exit(2)
    else:
        print('hash matches: good to go!')

    return


def create_connection(db_file=db_path + db_name) -> dict:
    """
    Create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    db = {'con': None, 'cur': None}
    try:
        db['con'] = sq.connect(db_file)
        db['cur'] = db['con'].cursor()
    except sq.Error as e:
        print(e)
    return db


def close_connection(db_conn: sq.Connection) -> None:
    return db_conn.close()


if __name__ == "__main__":
    # change the working directory to the Jupyter_Notebooks directory which is the same context the
    # the Jupyter Notebooks and foundational iPython are executing from; this enables the relative pathing to work the same
    os.chdir( '..')
    print('Working Directory:\t' + str(os.getcwd()))
    get_and_check_db_file()
