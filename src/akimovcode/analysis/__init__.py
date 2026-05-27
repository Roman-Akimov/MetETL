import logging

import akimovcode.analysis.processor as p
import akimovcode.analysis.analysis_core as ac
import akimovcode.analysis.visual as vis
import akimovcode.analysis.analysis_metrics as am

logger = logging.getLogger(__name__)


def main():
    file_path = "data/MetObjects.csv"

    logger.info("Запуск анализа")

    raw_stream = p.read_csv(file_path)
    filtered_stream = p.filtered_data(raw_stream)
    clean_stream = p.clean_data(filtered_stream)

    logger.info("Расчет статистики по отделам")

    stats, dates_storage = ac.calculate_statistics(clean_stream)

    logger.info("\n%s", stats.head())

    vis.draw_department_chart(
        stats,
        "data/plots/1_departments_stats.png"
    )

    logger.info("Построение временного графика")

    target_dept, timeline_data = (
        ac.get_max_spread_department(stats, dates_storage)
    )

    vis.draw_timeline(
        timeline_data,
        target_dept,
        "data/plots/2_timeline.png"
    )

    logger.info("Анализ категорий")

    metrics_stream = am.read_csv_with_categories(file_path)
    total_counts = am.sum_counts(metrics_stream)
    metrics_results = am.compute_metrics(total_counts)

    am.draw_heatmap(metrics_results)

    logger.info("Анализ завершен")
