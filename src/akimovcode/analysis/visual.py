import matplotlib.pyplot as plt
import pandas as pd


def draw_department_chart(stats, save_path):
    plt.figure(figsize=(14, 8))

    plt.bar(stats.index, stats["mean"], color="steelblue", label="Средний год")
    plt.errorbar(stats.index, stats["mean"], yerr=stats["ci_95"], fmt="none", ecolor="red", capsize=5, label="95% CI")

    for i, (idx, row) in enumerate(stats.iterrows()):
        low = row["mean"] - row["spread_95"]
        high = row["mean"] + row["spread_95"]
        plt.hlines([low, high], i - 0.25, i + 0.25, colors="green", linewidth=2,
                   label="Интервал рассеяния" if i == 0 else "")

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Год создания")
    plt.title("Статистика по отделам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_timeline(dates, dept_name, save_path):
    dates = pd.to_numeric(dates, errors="coerce").dropna()
    if dates.empty:
        return

    dates = dates[(dates >= 500) & (dates <= 2025)]

    if dates.empty:
        return

    counts = dates.value_counts().sort_index()

    window = max(5, len(counts) // 20)
    rolling = counts.rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(counts.index, counts.values, alpha=0.5, linewidth=1, label="Объекты")
    plt.plot(rolling.index, rolling.values, "r-", linewidth=2.5, label=f"Скользящее среднее (окно {window})")

    plt.xlabel("Год")
    plt.ylabel("Количество объектов")
    plt.title(f"{dept_name} (c 500 г. н.э.)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
