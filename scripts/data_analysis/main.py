import processor as dp
import analysis_core as ac
import visual as vis
import analysis_metrics as am


def main():
    file_path = "data/MetObjects.csv"

    print("Запуск")
    raw_stream = dp.read_csv(file_path)
    filtered_stream = dp.filtered_data(raw_stream)
    clean_stream = dp.clean_data(filtered_stream)

    print("Расчет статистики по отделам...")
    stats = ac.calculate_statistics(clean_stream)
    print(stats.head())

    vis.draw_department_chart(stats, "scripts/data_analysis/results/1_departments_stats.png")

    print("Поиск отдела с максимальным разбросом и сбор данных для графика...")
    raw_stream_v2 = dp.read_csv(file_path)
    filtered_stream_v2 = dp.filtered_data(raw_stream_v2)
    clean_stream_v2 = dp.clean_data(filtered_stream_v2)

    target_dept, timeline_data = ac.get_max_spread_department(stats, clean_stream_v2)
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
