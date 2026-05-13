import pandas as pd
import numpy as np


def calculate_statistics(chunks):
    full_depts = None
    dates_storage = {}

    for data in chunks:
        data["Date_Sq"] = data["Date"] ** 2

        grouped = data.groupby("Department")

        for dept, group in grouped:
            if dept not in dates_storage:
                dates_storage[dept] = []
            dates_storage[dept].extend(group["Date"].tolist())

        group_stats = grouped.agg(
            sum_date=("Date", "sum"),
            sum_sq=("Date_Sq", "sum"),
            count=("Date", "size")
        )

        if full_depts is None:
            full_depts = group_stats
        else:
            full_depts = full_depts.add(group_stats, fill_value=0)

    full_depts["mean"] = full_depts["sum_date"] / full_depts["count"]

    numerator = full_depts["sum_sq"] - (full_depts["sum_date"] ** 2 / full_depts["count"])
    denominator = full_depts["count"] - 1

    full_depts["std"] = np.sqrt(numerator / denominator)

    full_depts["ci_95"] = 1.96 * (full_depts["std"] / np.sqrt(full_depts["count"]))
    full_depts["spread_95"] = 1.96 * full_depts["std"]

    return full_depts, dates_storage


def get_max_spread_department(stats, dates_storage):
    if stats.empty:
        return None, pd.Series(dtype=float)

    target_dept = stats["std"].idxmax()

    if target_dept not in dates_storage:
        return target_dept, pd.Series(dtype=float)

    dates = pd.Series(dates_storage[target_dept]).sort_values()

    return target_dept, dates
