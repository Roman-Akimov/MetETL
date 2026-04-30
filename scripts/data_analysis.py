import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Установка стиля оформления
plt.style.use('ggplot')
warnings.filterwarnings('ignore')

def chunk_reader(file_path, chunksize=20000):
    """Генератор для чтения файла по частям."""
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        yield chunk

def filter_chunk(chunks):
    """Генератор фильтрации: замена полей под Вариант 1 (Department, Object Begin Date)."""
    for chunk in chunks:
        # Используем поля Department и Object Begin Date
        required_fields = ['Department', 'Object Begin Date']
        
        # Проверка наличия и очистка от NaN
        for field in required_fields:
            if field in chunk.columns:
                chunk = chunk[chunk[field].notna()]

        if 'Object Begin Date' in chunk.columns:
            chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
            chunk = chunk[chunk['Object Begin Date'].notna()]
            # Ограничим разумными историческими рамками
            chunk = chunk[chunk['Object Begin Date'].between(-5000, 2026)]

        if len(chunk) > 0:
            yield chunk

def process_chunk_for_analysis(chunks):
    """Генератор подготовки данных для Варианта 1."""
    for chunk in chunks:
        # Нам нужен только отдел и дата начала создания
        if 'Department' in chunk.columns and 'Object Begin Date' in chunk.columns:
            yield chunk[['Department', 'Object Begin Date']].copy()

def collect_data(chunks):
    """Сбор данных из генератора в один DataFrame для финальной агрегации."""
    all_data = []
    for chunk in chunks:
        all_data.append(chunk)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def calculate_department_stats(df):
    """Расчет среднего, CI и интервала рассеяния для каждого отдела."""
    results = []
    # Группировка по Department
    for dept, group in df.groupby('Department'):
        dates = group['Object Begin Date']
        n = len(dates)
        if n < 2: continue

        mean_val = dates.mean()
        std_val = dates.std()
        
        # 95% Доверительный интервал (CI)
        ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean_val, scale=std_val/np.sqrt(n))
        
        # 95% Интервал рассеяния (PI) - через квантили
        scatter_lower = dates.quantile(0.025)
        scatter_upper = dates.quantile(0.975)

        results.append({
            'Department': dept,
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'scatter_lower': scatter_lower,
            'scatter_upper': scatter_upper,
            'count': n
        })
    return pd.DataFrame(results)

def plot_department_bars(stats_df):
    """Построение столбчатой диаграммы"""
    stats_df = stats_df.sort_values('mean')
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(stats_df))
    # Основные бары (среднее значение)
    plt.bar(x, stats_df['mean'], color='skyblue', label='Средняя дата создания', alpha=0.7)
    
    # Ошибки: Доверительный интервал (черные усы)
    plt.errorbar(x, stats_df['mean'], 
                 yerr=[stats_df['mean'] - stats_df['ci_lower'], stats_df['ci_upper'] - stats_df['mean']], 
                 fmt='none', ecolor='black', capsize=5, label='95% CI')
    
    # Интервал рассеяния (заливка фоном)
    plt.fill_between(x, stats_df['scatter_lower'], stats_df['scatter_upper'], 
                     color='orange', alpha=0.2, label='95% Интервал рассеяния')

    plt.xticks(x, stats_df['Department'], rotation=90)
    plt.ylabel('Год создания (Object Begin Date)')
    plt.title('Анализ возраста объектов по отделам (Вариант 1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dept_analysis_bars.png')
    plt.show()

def plot_max_scatter_temporal(df, stats_df):
    """Временной график для отдела с макс. разбросом (Пункт 1.2)."""
    # Поиск отдела с максимальным стандартным отклонением (разбросом)
    target_dept = stats_df.loc[stats_df['std'].idxmax(), 'Department']
    dept_data = df[df['Department'] == target_dept].sort_values('Object Begin Date')
    
    dates = dept_data['Object Begin Date'].values
    # Скользящее среднее
    window = max(10, len(dates) // 20)
    rolling_mean = pd.Series(dates).rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(dates)), dates, alpha=0.3, s=5, label='Точки данных (объекты)')
    plt.plot(range(len(dates)), rolling_mean, color='red', linewidth=2, label=f'Скользящее среднее (окно {window})')
    
    plt.title(f'Временной график: {target_dept}\n(Отдел с макс. разбросом дат)')
    plt.xlabel('Порядковый номер объекта (отсортировано по времени)')
    plt.ylabel('Год создания')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('max_scatter_temporal.png')
    plt.show()

def main():
    file_path = 'data/MetObjects.csv'
    
    print("Запуск пайплайна обработки...")
    # Цепочка генераторов
    raw_chunks = chunk_reader(file_path)
    filtered_chunks = filter_chunk(raw_chunks)
    processed_chunks = process_chunk_for_analysis(filtered_chunks)
    
    # Сбор данных (один проход по генераторам)
    full_df = collect_data(processed_chunks)
    if full_df.empty:
        print("Данные не загружены. Проверьте путь к файлу.")
        return

    print(f"Обработано объектов: {len(full_df)}")

    # 1. Расчет статистик по отделам
    stats_df = calculate_department_stats(full_df)
    
    # 2. Вывод результатов в консоль
    print("\nСтатистика по отделам:")
    print(stats_df[['Department', 'mean', 'std', 'count']].to_string(index=False))

    # 3. Визуализация
    plot_department_bars(stats_df)
    plot_max_scatter_temporal(full_df, stats_df)

if __name__ == "__main__":
    main()