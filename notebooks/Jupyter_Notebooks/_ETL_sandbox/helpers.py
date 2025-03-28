import sklearn.linear_model as linear
import sqlite3
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import patsy
import pandas as pd
import numpy as np
from tabulate import tabulate

ALGORITHMS = {
    "linear": linear.LinearRegression,
    "ridge": linear.Ridge,
    "lasso": linear.Lasso
}


def create_connection(db_file: str) -> dict:
    """
    Create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    db = {'con': None, 'cur': None}
    try:
        db['con'] = sqlite3.connect(db_file)
        db['cur'] = db['con'].cursor()
    except sqlite3.Error as e:
        print(e)
    return db


def close_connection(db_conn: sqlite3.Connection) -> None:
    return db_conn.close()


def freeman_diaconis(data):
    quartiles = stats.mstats.mquantiles(data, [0.25, 0.5, 0.75])
    iqr = quartiles[2] - quartiles[0]
    n = len(data)
    h = 2.0 * (iqr / n ** (1.0 / 3.0))
    return int(h)


def plot_hist_categorical(t_col: pd.Series, do_normalize: bool):
    data = t_col.value_counts(normalize=do_normalize)
    x = list(data.index.sort_values())
    width = 1 / 1.5
    figure = plt.figure(figsize=(8, 6))

    axes = figure.add_subplot(1, 1, 1)
    axes.bar(x, data, width, align="center", color="darkslategray")
    axes.set_xticks(x)
    axes.set_xticklabels(data.axes[0])
    axes.set_title(' '.join(['Distribution of', t_col.name.title()]))
    axes.set_xlabel(t_col.name.title())
    axes.set_ylabel('Percent' if do_normalize else 'Count')
    axes.xaxis.grid(False)

    plt.show()
    plt.close()
    return


def plot_hist_numeric(t_col: pd.Series, backup_step=2, multiplier_factor=1):
    mn = int(t_col.min())
    mx = int(t_col.max())
    h = freeman_diaconis(t_col) * multiplier_factor
    if h == 0: h = backup_step
    bins = [i for i in range(mn, mx, h)]

    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    axes.hist(t_col, bins=bins, color="darkslategray")
    axes.set_title(' '.join([t_col.name.title(), 'Distribution']))
    axes.set_xlabel(t_col.name.title())

    plt.show()
    plt.close()
    return


def plot_hist_numeric_custom(t_col: pd.Series, bins: list):
    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    axes.hist(t_col, bins=bins, color="darkslategray")
    axes.set_title(' '.join([t_col.name.title(), 'Distribution']))
    axes.set_xlabel(t_col.name.title())

    plt.show()
    plt.close()
    return


def get_correlations(df: pd.DataFrame, colA: str, colB: str) -> dict:
    results = {}
    results['pearson'] = stats.pearsonr(df[colA], df[colB])[0]
    results['spearman'] = stats.spearmanr(df[colA], df[colB])[0]
    return results


def get_correlations_en_masse(data, y, xs: list) -> pd.DataFrame:
    rs = []
    rhos = []
    for x in xs:
        r = stats.pearsonr(data[y], data[x])[0]
        rs.append(r)
        rho = stats.spearmanr(data[y], data[x])[0]
        rhos.append(rho)
    return pd.DataFrame({"feature": xs, "r": rs, "rho": rhos})



def describe_by_category(my_data: pd.DataFrame, numeric: str, categorical: str, transpose=False):
    t_grouped = my_data.groupby(categorical)
    t_grouped_y = t_grouped[numeric].describe()
    if transpose:
        print(t_grouped_y.transpose())
    else:
        print(t_grouped_y)
    return t_grouped


def plot_scatter(my_data: pd.DataFrame, y_col: str, x_col: str):
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter(y=my_data[y_col], x=my_data[x_col], marker='o', color='darkslategray')
    axes.set_ylabel(y_col.title())
    axes.set_xlabel(x_col.title())
    axes.set_title(' '.join([y_col, 'vs.', x_col]))

    plt.show()
    plt.close()
    return


def plot_by_category(my_data: pd.DataFrame, response_col: str, explanatory_col: str, relative: bool):
    n_cols = 3
    h = freeman_diaconis(my_data[response_col])
    grouped = my_data.groupby(explanatory_col)
    figure = plt.figure(figsize=(20, 6))

    n_rows = math.ceil(grouped.ngroups / n_cols)

    for plot_index, k in enumerate(grouped.groups.keys()):
        axes = figure.add_subplot(n_rows, n_cols, plot_index + 1)
        axes.hist(grouped[response_col].get_group(k), bins=h, color="darkslategray", density=relative,
                  range=(0, 40))
        axes.set_title(
            ' '.join(
                [str(k), explanatory_col.title(), '-', response_col.title(), '\ndistribution - Freeman Diaconis']))
        axes.set_xlabel(response_col)

    figure.tight_layout()
    plt.show()
    plt.close()
    return

def linear_regression(formula, data=None, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    params["fit_intercept"] = False

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    algorithm = ALGORITHMS[style]
    algo = algorithm(**params)
    model = algo.fit(X, y)

    result = summarize(formula, X, y, model, style)

    return result

class ResultsWrapper(object):
    def __init__(self, fit, sd=2, bootstrap=False, is_logistic=False):
        self.fit = fit
        self.sd = sd
        self.bootstrap = bootstrap
        self.is_logistic = is_logistic

    def _repr_markdown_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="markdown")
        table = tabulate(table, tablefmt="github")
        markdown = title + "\n" + table
        return markdown

    def _repr_html_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="html")
        table = tabulate(table, tablefmt="html")
        table = table.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong")
        return f"<p><strong>{title}</strong><br/>{table}</p>"

    def _repr_latex_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="latex")

        title = title.replace("~", "$\\sim$").replace("_", "\\_")

        table = tabulate(table, tablefmt="latex_booktabs")
        table = table.replace("textbackslash{}", "").replace("\^{}", "^").replace("\_", "_")
        table = table.replace("\\$", "$").replace("\\{", "{").replace("\\}", "}")
        latex = "\\textbf{" + title + "}\n\n" + table
        return latex

def summarize(formula, X, y, model, style='linear'):
    result = {}
    result["formula"] = formula
    result["n"] = len(y)
    result["model"] = model
    # I think this is a bug in Scikit Learn
    # because lasso should work with multiple targets.
    if style == "lasso":
        result["coefficients"] = model.coef_
    else:
        result["coefficients"] = model.coef_[0]
    result["r_squared"] = model.score(X, y)
    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"] = y
    sum_squared_error = sum([e ** 2 for e in result["residuals"]])[0]

    n = len(result["residuals"])
    k = len(result["coefficients"])

    result["sigma"] = np.sqrt(sum_squared_error / (n - k))
    return result



def describe_bootstrap_lr(fit, sd=2):
    return ResultsWrapper(fit, sd, True, False)



def bootstrap_linear_regression(formula, data=None, samples=100, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split("+")]
    variables = ["intercept"] + variables
    bootstrap_results["variables"] = variables

    coeffs = []
    sigmas = []
    rs = []

    n = len(data)
    bootstrap_results["n"] = n

    for i in range(samples):
        sampling = data.sample(len(data), replace=True)
        results = linear_regression(formula, data=sampling, style=style, params=params)
        coeffs.append(results["coefficients"])
        sigmas.append(results["sigma"])
        rs.append(results["r_squared"])

    coeffs = pd.DataFrame(coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name="sigma")
    rs = pd.Series(rs, name="r_squared")

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_sigma"] = sigmas
    bootstrap_results["resampled_r^2"] = rs

    result = linear_regression(formula, data=data)

    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["sigma"] = result["sigma"]
    bootstrap_results["r_squared"] = result["r_squared"]
    bootstrap_results["model"] = result["model"]
    bootstrap_results["y"] = result["y"]
    bootstrap_results["y_hat"] = result["y_hat"]
    return bootstrap_results