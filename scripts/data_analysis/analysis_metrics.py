import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

categories = ["Department", "Culture", "Medium", "Classification", "Country"]


def read_csv_with_categories(path, chunk_size=10000):
    reader = pd.read_csv(path, usecols=categories, chunksize=chunk_size)
    for part in reader:
        yield part.copy()


def sum_counts(chunks):
    combined = None

    for part in chunks:
        current = None

        for field in categories:
            freq = part[field].value_counts()
            freq = freq.to_frame(name=field)

            if current is None:
                current = freq
            else:
                current = current.join(freq, how="outer")

        current = current.fillna(0)

        if combined is None:
            combined = current
        else:
            combined = combined.add(current, fill_value=0)

    if combined is None:
        return pd.DataFrame()

    return combined.fillna(0)


def compute_metrics(total_counts):
    output = pd.DataFrame(index=categories, columns=["gini", "shannon", "enc"])

    for field in categories:
        values = total_counts[field].dropna()
        values = values[values > 0]

        if values.shape[0] == 0:
            output.loc[field] = [np.nan, np.nan, np.nan]
            continue

        probs = values / values.sum()

        g = 1 - (probs * probs).sum()
        s = -(probs * np.log(probs)).sum()
        e = np.exp(s)

        output.loc[field] = [g, s, e]

    return output


def draw_heatmap(metrics_raw):
    df = metrics_raw.copy()
    df.columns = ["Gini", "Shannon", "ENC"]
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.shape[0] == 0 or df.isna().all().all():
        return

    max_vals = df.max()

    for col in df.columns:
        if max_vals[col] != 0:
            df[col] = df[col] / max_vals[col]

    plt.figure(figsize=(8, 5))
    plt.imshow(df.values, aspect="auto", cmap="Reds", vmin=0, vmax=1, interpolation='nearest')

    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.index)), df.index)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/3_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
