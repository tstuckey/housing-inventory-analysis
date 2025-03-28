class Queries:
    main_query = """
WITH mortgage_monthly(year_month, mortgage_rate) AS (
    SELECT year_month, avg(mortgage_rate) from mortgage_rates
    group by year_month
)
SELECT  hi.total_listing_count as 'housing_inventory', bp.total_units as 'housing_permits',
        mm.mortgage_rate, pr.prime_rate, rc.credit, hi.cbsa_code, mm.year_month
FROM housing_inventory as hi
INNER JOIN building_permits bp
    on hi.year_month = bp.year_month and hi.cbsa_code = bp.cbsa_code
INNER JOIN mortgage_monthly mm
    on hi.year_month = mm.year_month
INNER JOIN prime_rates pr
    on hi.year_month = pr.year_month
INNER JOIN revolving_credit rc
    on hi.year_month = rc.year_month;
"""

    cbsa_query = """
   SELECT count(distinct(cbsa_code)) from housing_inventory;
    """


    cbsa_top3 = """
SELECT distinct(cbsa_title)
from housing_inventory
WHERE cbsa_code = 35620
   OR cbsa_code = 33100
   OR cbsa_code = 16980;
"""
