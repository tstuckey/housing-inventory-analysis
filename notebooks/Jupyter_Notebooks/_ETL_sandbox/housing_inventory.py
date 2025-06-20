import pandas as pd
import basic_helpers as hp

src_csv = '../_data/src data/RDC_Inventory_Core_Metrics_Metro_History.csv'
db_file = '../_data/housing_inventory.db'
main_columns = ['month_date_yyyymm', 'cbsa_code', 'cbsa_title', 'total_listing_count']


def get_info(t_file: str) -> pd.DataFrame:
    """
    Get the info out of the csv file and handle the year month breakout
    :param t_file:
    :return:
    """
    df = pd.read_csv(t_file, header=0)
    df = df.loc[:, df.columns.isin(main_columns)]
    return df


def fill_db(t_df: pd.DataFrame, db: dict) -> None:
    """
    Take the dictionary and fill the database
    """
    con = db['con']
    cur = db['cur']

    entries = []
    for index, row in t_df.iterrows():
        entries.append([row.month_date_yyyymm, row.cbsa_code, row.cbsa_title, row.total_listing_count])
    cur.executemany(
        'INSERT into housing_inventory (year_month, cbsa_code, cbsa_title, total_listing_count) VALUES (?, ?, ?, ?)',
        entries)
    con.commit()
    return


def main(t_db_file=db_file):
    housing_inventory_df = get_info(src_csv)
    db_ref = hp.create_connection(t_db_file)
    fill_db(housing_inventory_df, db_ref)
    hp.close_connection(db_ref['con'])
    return


if __name__ == '__main__':
    main()
