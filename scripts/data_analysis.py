import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

plt.style.use('ggplot')
warnings.filterwarnings('ignore')

def chunk_reader(file_path, chunksize=20000):
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        yield chunk

def filter_chunk(chunks):
    for chunk in chunks:
        required_fields = ['Department', 'Object Begin Date']
        
        for field in required_fields:
            if field in chunk.columns:
                chunk = chunk[chunk[field].notna()]

        if 'Object Begin Date' in chunk.columns:
            chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
            chunk = chunk[chunk['Object Begin Date'].notna()]
            chunk = chunk[chunk['Object Begin Date'].between(-5000, 2026)]

        if len(chunk) > 0:
            yield chunk

def process_chunk_for_analysis(chunks):
    for chunk in chunks:
        if 'Department' in chunk.columns and 'Object Begin Date' in chunk.columns:
            yield chunk[['Department', 'Object Begin Date']].copy()

def collect_data(chunks):
    all_data = []
    for chunk in chunks:
        all_data.append(chunk)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def calculate_department_stats(df):
    results = []
    for dept, group in df.groupby('Department'):
        dates = group['Object Begin Date']
        n = len(dates)
        if n < 2: continue

        mean_val = dates.mean()
        std_val = dates.std()
        
        ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean_val, scale=std_val/np.sqrt(n))
        
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
    stats_df = stats_df.sort_values('mean')
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(stats_df))
    plt.bar(x, stats_df['mean'], color='skyblue', alpha=0.7)
    plt.errorbar(x, stats_df['mean'], 
                 yerr=[stats_df['mean'] - stats_df['ci_lower'], stats_df['ci_upper'] - stats_df['mean']], 
                 fmt='none', ecolor='black', capsize=5)
    plt.fill_between(x, stats_df['scatter_lower'], stats_df['scatter_upper'], 
                     color='orange', alpha=0.2)

    plt.xticks(x, stats_df['Department'], rotation=90)
    plt.ylabel('Year created')
    plt.title('Department analysis')
    plt.legend(['Mean', '95% CI', '95% scatter interval'])
    plt.tight_layout()
    plt.savefig('dept_analysis_bars.png')
    plt.show()

def plot_max_scatter_temporal(df, stats_df):
    target_dept = stats_df.loc[stats_df['std'].idxmax(), 'Department']
    dept_data = df[df['Department'] == target_dept].sort_values('Object Begin Date')
    
    dates = dept_data['Object Begin Date'].values
    window = max(10, len(dates) // 20)
    rolling_mean = pd.Series(dates).rolling(window=window, center=True).mean()

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(dates)), dates, alpha=0.3, s=5)
    plt.plot(range(len(dates)), rolling_mean, color='red', linewidth=2)
    
    plt.title(f'Temporal plot: {target_dept}')
    plt.xlabel('Object index (sorted by date)')
    plt.ylabel('Year created')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('max_scatter_temporal.png')
    plt.show()

def calculate_gini(data):
    counts = data.value_counts()
    n = len(data)
    if n == 0:
        return 0
    
    probs = counts / n
    gini = 1 - np.sum(probs ** 2)
    max_gini = 1 - 1/len(counts) if len(counts) > 1 else 0
    gini_norm = gini / max_gini if max_gini > 0 else 0
    
    return gini_norm

def calculate_shannon_entropy(data):
    counts = data.value_counts()
    n = len(data)
    if n == 0:
        return 0
    
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs))
    
    k = len(counts)
    max_entropy = np.log2(k) if k > 1 else 1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def calculate_enc(data):
    counts = data.value_counts()
    n = len(data)
    if n == 0:
        return 0
    
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs))
    enc = 2 ** entropy
    
    return enc

def analyze_field_quality(df, field_name):
    if field_name not in df.columns:
        return None
    
    data = df[field_name].dropna()
    if len(data) == 0:
        return None
    
    n_total = len(data)
    n_unique = data.nunique()
    n_missing = df[field_name].isna().sum()
    top_values = data.value_counts().head(5)
    
    gini = calculate_gini(data)
    entropy = calculate_shannon_entropy(data)
    enc = calculate_enc(data)
    
    issues = []
    
    if gini < 0.3:
        issues.append("Low diversity")
    elif gini > 0.8:
        issues.append("Very high diversity")
    
    if entropy < 0.3:
        issues.append("Low entropy")
    elif entropy > 0.9:
        issues.append("High entropy")
    
    if n_unique > 100:
        issues.append(f"Too many categories ({n_unique})")
    elif n_unique < 3 and n_total > 100:
        issues.append("Almost constant field")
    
    low_quality_indicators = ['unknown', 'unidentified', 'various', 'other', 'uncertain', 'probably']
    if data.dtype == 'object':
        low_quality_count = data.str.lower().str.contains('|'.join(low_quality_indicators), na=False).sum()
        if low_quality_count > n_total * 0.1:
            issues.append(f"Many uncertain values ({low_quality_count}/{n_total})")
    
    return {
        'field': field_name,
        'total_count': n_total,
        'missing_count': n_missing,
        'missing_pct': n_missing / (n_total + n_missing) * 100,
        'unique_categories': n_unique,
        'gini_coefficient': gini,
        'normalized_entropy': entropy,
        'effective_categories': enc,
        'top1_value': top_values.index[0] if len(top_values) > 0 else None,
        'top1_pct': (top_values.iloc[0] / n_total * 100) if len(top_values) > 0 else 0,
        'issues': ', '.join(issues) if issues else 'Good quality',
        'quality_score': (gini + entropy + (enc / n_unique if n_unique > 0 else 0)) / 3
    }

def plot_quality_heatmap(quality_results):
    metrics_df = pd.DataFrame(quality_results)
    metrics_df = metrics_df.set_index('field')
    
    heatmap_data = metrics_df[['gini_coefficient', 'normalized_entropy', 'missing_pct', 'quality_score']]
    heatmap_data['missing_pct'] = heatmap_data['missing_pct'] / 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5, vmin=0, vmax=1)
    
    plt.title('Quality heatmap of categorical fields', fontsize=14, fontweight='bold')
    plt.ylabel('Categorical fields')
    plt.xlabel('Quality metrics')
    plt.tight_layout()
    plt.savefig('quality_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

def suggest_normalization_rules(quality_results):
    suggestions = []
    
    for result in quality_results:
        if result['quality_score'] > 0.7:
            continue
        
        field = result['field']
        issues = result['issues']
        
        suggestion = {
            'field': field,
            'current_quality': result['quality_score'],
            'issues_identified': issues,
            'normalization_rules': []
        }
        
        if result['unique_categories'] > 50:
            if field == 'Culture':
                suggestion['normalization_rules'].extend([
                    "Combine spelling variants",
                    "Group by region",
                    "Merge related categories"
                ])
            elif field == 'Medium':
                suggestion['normalization_rules'].extend([
                    "Normalize material names",
                    "Group by type",
                    "Extract main components"
                ])
            else:
                suggestion['normalization_rules'].append("Group into semantic clusters")
        
        if 'uncertain' in issues.lower():
            suggestion['normalization_rules'].extend([
                "Replace 'unknown' with NaN",
                "Cross-validate uncertain values",
                "Mark probable values as valid"
            ])
        
        if result['gini_coefficient'] < 0.3:
            suggestion['normalization_rules'].append("Consider removing this field from analysis")
        
        if result['missing_pct'] > 20:
            suggestion['normalization_rules'].append("Handle high missing rate: drop, impute, or keep as separate category")
        
        if suggestion['normalization_rules']:
            potential_improvement = (1 - result['quality_score']) * 0.7
            suggestion['potential_improvement'] = min(0.95, result['quality_score'] + potential_improvement)
            suggestion['improvement_description'] = f"Can be improved from {result['quality_score']:.3f} to {suggestion['potential_improvement']:.3f}"
        else:
            suggestion['potential_improvement'] = result['quality_score']
            suggestion['improvement_description'] = "Already good quality"
        
        suggestions.append(suggestion)
    
    return suggestions

def print_quality_report(quality_results, suggestions):
    print("\n" + "="*80)
    print("Categorical fields quality report")
    print("="*80)
    
    quality_results_sorted = sorted(quality_results, key=lambda x: x['quality_score'])
    
    for result in quality_results_sorted:
        print(f"\nField: {result['field']}")
        print(f"   Total: {result['total_count']}")
        print(f"   Missing: {result['missing_count']} ({result['missing_pct']:.1f}%)")
        print(f"   Unique categories: {result['unique_categories']}")
        print(f"   Gini coefficient: {result['gini_coefficient']:.4f}")
        print(f"   Normalized entropy: {result['normalized_entropy']:.4f}")
        print(f"   Effective categories: {result['effective_categories']:.2f}")
        print(f"   Most frequent: '{result['top1_value']}' ({result['top1_pct']:.1f}%)")
        print(f"   Issues: {result['issues']}")
        print(f"   Quality score: {result['quality_score']:.3f}")
    
    print("\n" + "="*80)
    print("Normalization suggestions")
    print("="*80)
    
    for suggestion in suggestions:
        print(f"\n{suggestion['field']}")
        print(f"   Current quality: {suggestion['current_quality']:.3f}")
        print(f"   Issues: {suggestion['issues_identified']}")
        print(f"   Rules:")
        for rule in suggestion['normalization_rules']:
            print(f"      - {rule}")
        print(f"   {suggestion['improvement_description']}")
    
    print("\n" + "="*80)
    print("Final recommendations")
    print("="*80)
    
    avg_quality = np.mean([r['quality_score'] for r in quality_results])
    print(f"Average quality: {avg_quality:.3f}")
    
    if avg_quality < 0.4:
        print("Serious normalization required")
    elif avg_quality < 0.6:
        print("Normalization recommended")
    else:
        print("Acceptable quality")
    
    poor_fields = [r['field'] for r in quality_results if r['quality_score'] < 0.5]
    if poor_fields:
        print(f"Priority fields: {', '.join(poor_fields)}")

def main_advanced_analysis(df):
    print("\nRunning quality analysis")
    
    fields_to_analyze = ['Department', 'Culture', 'Medium', 'Classification', 'Country']
    
    quality_results = []
    for field in fields_to_analyze:
        if field in df.columns:
            result = analyze_field_quality(df, field)
            if result:
                quality_results.append(result)
    
    plot_quality_heatmap(quality_results)
    suggestions = suggest_normalization_rules(quality_results)
    print_quality_report(quality_results, suggestions)
    
    return quality_results, suggestions

def main():
    file_path = 'data/MetObjects.csv'
    
    print("Running pipeline")
    raw_chunks = chunk_reader(file_path)
    filtered_chunks = filter_chunk(raw_chunks)
    processed_chunks = process_chunk_for_analysis(filtered_chunks)
    
    full_df = collect_data(processed_chunks)
    if full_df.empty:
        print("No data loaded")
        return

    print(f"Processed objects: {len(full_df)}")

    stats_df = calculate_department_stats(full_df)
    
    print("\nDepartment statistics:")
    print(stats_df[['Department', 'mean', 'std', 'count']].to_string(index=False))

    plot_department_bars(stats_df)
    plot_max_scatter_temporal(full_df, stats_df)
    
    quality_results, suggestions = main_advanced_analysis(full_df)
    
    quality_df = pd.DataFrame(quality_results)
    quality_df.to_csv('quality_analysis_report.csv', index=False)
    print("\nReport saved to quality_analysis_report.csv")

if __name__ == "__main__":
    main()