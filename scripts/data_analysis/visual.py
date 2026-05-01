import matplotlib.pyplot as plt


def draw_department_chart(stats, save_path):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(14, 8))

    plt.bar(stats.index, stats["mean"], color="#7fb3d5", edgecolor="#2e4053", linewidth=0.8, label="Средний год",
            alpha=0.85)

    plt.errorbar(stats.index, stats["mean"], yerr=stats["ci_95"],
                 fmt="none", ecolor="#c0392b", elinewidth=1.5, capsize=4,
                 label="95% CI")

    for i, (idx, row) in enumerate(stats.iterrows()):
        spread_low = row["mean"] - row["spread_95"]
        spread_high = row["mean"] + row["spread_95"]

        plt.hlines([spread_low, spread_high], i - 0.25, i + 0.25,
                   colors="#1e8449", linestyles='solid', linewidth=2.5,
                   label="Интервал рассеяния" if i == 0 else "")

    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.gca().set_axisbelow(True)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Год начала создания", fontsize=12, fontweight='bold')
    plt.title("Статистика возраста объектов по отделам музея", fontsize=14, pad=20, fontweight='bold')

    plt.legend(frameon=True, shadow=True, loc='best')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_timeline(dates, dept_name, save_path):
    if dates.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    counts = dates.value_counts().sort_index()

    window = max(5, len(counts) // 10)
    rolling = counts.rolling(window=window, center=True).mean()

    plt.fill_between(counts.index, counts.values, 0, alpha=0.2, color='#4A90E2')

    plt.plot(counts.index, counts.values, alpha=0.5, color='#4A90E2', linewidth=1)
    plt.plot(rolling.index, rolling.values, color='#E74C3C', linewidth=2.5)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.grid(axis='y', linestyle='-', alpha=0.15)
    plt.title(dept_name, fontsize=16, pad=20, color='#2C3E50')
    plt.xlabel('Year', fontsize=11, color='#7F8C8D')
    plt.ylabel('Number of objects', fontsize=11, color='#7F8C8D')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
