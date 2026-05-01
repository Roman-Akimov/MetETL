import pandas as pd
import numpy as np


def calculate_statistics(chunks):
    full_depts = None

    for data in chunks:
        data["Date_Sq"] = data["Date"] ** 2

        group = data.groupby("Department").agg(
            sum_date=("Date", "sum"),
            sum_sq=("Date_Sq", "sum"),
            count=("Date", "size"))

        if full_depts is None:
            full_depts = group
        else:
            full_depts = full_depts.add(group, fill_value=0)

    full_depts["mean"] = full_depts["sum_date"] / full_depts["count"]

    numerator = full_depts["sum_sq"] - (full_depts["sum_date"]**2 / full_depts["count"])
    denominator = full_depts["count"] - 1
    full_depts["std"] = np.sqrt(numerator / denominator)
    full_depts["ci_95"] = 1.96 * (full_depts["std"] / np.sqrt(full_depts["count"]))
    full_depts["spread_95"] = 1.96 * full_depts["std"]

    return full_depts


def get_max_spread_department(stats, stream):
    target_dept = stats["std"].idxmax()
    all_dates = []

    for data in stream:
        filtered = data[data["Department"] == target_dept]["Date"]
        all_dates.append(filtered)

    return target_dept, pd.concat(all_dates).sort_values()
