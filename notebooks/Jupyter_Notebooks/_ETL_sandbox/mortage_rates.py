import openpyxl as op
import basic_helpers as hp
from datetime import datetime

master_tab = 'Full History'

src_file = '../_data/src data/historical weekly mortgage data.xlsx'
db_file = '../_data/housing_inventory.db'


def get_info(t_file: str) -> list:
    """
    Get the info out of the csv file and handle the year month breakout
    :param t_file:
    :return:
    """
    wb = op.load_workbook(t_file)
    results=[]
    master_sheet = wb[master_tab]
    for cell_value in master_sheet.iter_rows(min_row=8, min_col=0, max_col=2, values_only=True):
        if cell_value[0] is None: break
        t_date = cell_value[0]
        new_date = datetime.strftime(t_date,'%Y%m')
        # only retain the records after June 2016 for import into the db
        if new_date > '201606':
            results.append([new_date, cell_value[1]])
    return results


def fill_db(t_entries: list, db: dict) -> None:
    """
    Take the dictionary and fill the database
    """
    con = db['con']
    cur = db['cur']

    cur.executemany(
        'INSERT into mortgage_rates (year_month, mortgage_rate) VALUES (?, ?)',
        t_entries)
    con.commit()
    return


def main(t_db_file = db_file):
    mortgage_inventory = get_info(src_file)
    db_ref = hp.create_connection(t_db_file)
    fill_db(mortgage_inventory, db_ref)
    hp.close_connection(db_ref['con'])
    return


if __name__ == '__main__':
    main()
