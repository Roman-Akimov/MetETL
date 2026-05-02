import processor as p
import analysis_core as ac
import visual as vis
import analysis_metrics as am


def main():
    file_path = "data/MetObjects.csv"

    print("Запуск")
    raw_stream = p.read_csv(file_path)
    filtered_stream = p.filtered_data(raw_stream)
    clean_stream = p.clean_data(filtered_stream)

    print("Расчет статистики по отделам...")
    stats, dates_storage = ac.calculate_statistics(clean_stream)
    print(stats.head())

    vis.draw_department_chart(stats, "scripts/data_analysis/results/1_departments_stats.png")

    print("Поиск отдела с максимальным разбросом и построение графика...")
    target_dept, timeline_data = ac.get_max_spread_department(stats, dates_storage)

    vis.draw_timeline(timeline_data, target_dept, "scripts/data_analysis/results/2_timeline.png")

    print("\nДопы")
    print("Анализ полей...")
    metrics_stream = am.read_csv_with_categories(file_path)
    total_counts = am.sum_counts(metrics_stream)
    metrics_results = am.compute_metrics(total_counts)

    am.draw_heatmap(metrics_results)
    print("Результаты в папке results.")


if __name__ == "__main__":
    main()
