"""
Anytime Algorithm Selection Benchmark across Multiple Random Seeds (10 Seeds)
=============================================================================
This script evaluates the Anytime Algorithm Selection (AAS) framework for CVRP
across 10 distinct random seeds. It features:
  - Automatic data caching (preprocess once into a fast pickle cache; subsequent runs load in <0.2s)
  - Strictly instance-wise 80/20 train/test split per seed
  - Training of 5 Machine Learning models (Gradient Boosting Classifier, Random Forest,
    MLP, Gradient Boosting Regressor, LightGBM Ranker)
  - Evaluation against baselines (Single Best Solver, Uniform Random, Virtual Best Solver)
  - Statistical aggregation across 10 seeds (Mean ± Standard Deviation)
  - Exporting results to Excel and CSV summary files
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Default 10 Evaluation Seeds
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_DATA_PATH = 'or-tools_gaetano_benchmark_(1,000_random_instances)_20501208_174656.xlsx'
DEFAULT_CACHE_PATH = 'anytime_cvrp_preprocessed_cache.pkl'


class DualLogger:
    """Redirects stdout to both console and a log file simultaneously."""
    def __init__(self, filepath: str, mode: str = 'w', encoding: str = 'utf-8'):
        self.terminal = sys.stdout
        self.log_file = open(filepath, mode=mode, encoding=encoding)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

    def close(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()


def abbreviate_alg(name: str) -> str:
    """Utility method to format verbose heuristic and metaheuristic names."""
    return (name.replace('Parallel Cheapest Insertion', 'PCI')
                .replace('Guided Local Search (GLS)', 'GLS')
                .replace('Tabu Search', 'TS')
                .replace('Simulated Annealing', 'SA')
                .replace('Savings (Clarke-Wright)', 'Savings')
                .replace('_', ' + '))


def load_or_preprocess_data(data_path: str = DEFAULT_DATA_PATH, cache_path: str = DEFAULT_CACHE_PATH, force_reload: bool = False):
    """
    Loads preprocessed anytime CVRP datasets from a pickle cache if available.
    If not, parses the raw Excel benchmark data, extracts all time steps (0.5s - 100s),
    computes tying winners, and saves to the cache for instantaneous loading in subsequent runs.
    """
    if os.path.exists(cache_path) and not force_reload:
        print(f"Loading preprocessed dataset from cache: '{cache_path}'...")
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Cache loaded successfully in {time.time() - t0:.2f}s!")
        return cache_data['df_decisions'], cache_data['df_regression'], cache_data['alg_names'], cache_data['feature_cols'], cache_data['categorical_cols'], cache_data['numerical_cols']

    print(f"Reading raw benchmark Excel file from '{data_path}' (this is done only once)...")
    t0 = time.time()
    raw_df = pd.read_excel(data_path)
    print(f"Excel read in {time.time() - t0:.2f}s. Total rows: {len(raw_df):,}")

    raw_df['Algorithm'] = raw_df['First Solution'] + '_' + raw_df['Metaheuristic']
    alg_names = sorted(raw_df['Algorithm'].unique().tolist())

    feature_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate', 'Customers', 'Vehicles', 'Capacity']
    categorical_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate']
    numerical_cols = ['Customers', 'Vehicles', 'Capacity', 'Time_Budget']

    print("Structuring anytime decision matrices and regression records across 1,000 instances...")
    instance_to_features = raw_df.drop_duplicates(subset=['Instance']).set_index('Instance')[feature_cols]
    instance_to_bks = raw_df.drop_duplicates(subset=['Instance']).set_index('Instance')['BKS Cost']
    time_steps = np.arange(0.5, 100.5, 0.5)

    classification_data = []
    regression_data = []

    grouped = raw_df.groupby('Instance')
    for instance, group in grouped:
        features = instance_to_features.loc[instance].to_dict()
        bks = instance_to_bks.loc[instance]
        n_customers = features['Customers']
        max_t = 0.5 * n_customers
        valid_ts = [t for t in time_steps if t <= max_t]

        alg_costs = {}
        for _, row in group.iterrows():
            alg = row['Algorithm']
            alg_costs[alg] = {}
            for t in valid_ts:
                col_name = f'Avg Cost @ {t:g}s'
                if col_name not in row or pd.isna(row[col_name]):
                    col_name = f'Best Cost @ {t:g}s'
                
                if col_name in row and pd.notna(row[col_name]):
                    alg_costs[alg][t] = row[col_name]
                else:
                    alg_costs[alg][t] = float('inf')

        for t in valid_ts:
            valid_costs = {alg: costs[t] for alg, costs in alg_costs.items() if costs[t] != float('inf')}
            if not valid_costs:
                continue

            min_cost = min(valid_costs.values())
            tying_algs = [alg for alg, c in valid_costs.items() if np.isclose(c, min_cost, rtol=1e-5, atol=1e-2)]

            # Regression rows
            for alg, cost in valid_costs.items():
                reg_row = features.copy()
                reg_row['Instance'] = instance
                reg_row['Time_Budget'] = t
                reg_row['Algorithm'] = alg
                reg_row['Cost'] = cost
                reg_row['BKS'] = bks
                regression_data.append(reg_row)

            # Decision row
            class_row = features.copy()
            class_row['Instance'] = instance
            class_row['Time_Budget'] = t
            class_row['BKS'] = bks
            class_row['Tying_Winners'] = tying_algs
            class_row['Num_Winners'] = len(tying_algs)
            class_row['Best_Cost'] = min_cost
            for alg in alg_names:
                class_row[f'Cost_{alg}'] = alg_costs[alg].get(t, float('inf'))
            classification_data.append(class_row)

    df_decisions = pd.DataFrame(classification_data)
    df_regression = pd.DataFrame(regression_data)

    print(f"Decisions dataset: {len(df_decisions):,} rows across 1,000 instances")
    print(f"Regression dataset: {len(df_regression):,} samples")

    print(f"Saving preprocessed datasets to fast cache '{cache_path}'...")
    cache_payload = {
        'df_decisions': df_decisions,
        'df_regression': df_regression,
        'alg_names': alg_names,
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cache saved successfully!")

    return df_decisions, df_regression, alg_names, feature_cols, categorical_cols, numerical_cols


def evaluate_single_seed(seed: int, df_decisions: pd.DataFrame, df_regression: pd.DataFrame,
                         alg_names: list, feature_cols: list, categorical_cols: list, numerical_cols: list):
    """
    Executes a complete Anytime Algorithm Selection benchmark run for a single random seed.
    Performs strictly instance-wise 80/20 train/test split, trains 5 ML models + 3 baselines,
    and returns comprehensive academic evaluation metrics.
    """
    t_start = time.time()
    unique_instances = df_decisions['Instance'].unique()
    train_instances, test_instances = train_test_split(unique_instances, test_size=0.2, random_state=seed)

    train_mask = df_decisions['Instance'].isin(train_instances)
    test_mask = df_decisions['Instance'].isin(test_instances)

    train_df = df_decisions[train_mask].reset_index(drop=True)
    test_df = df_decisions[test_mask].reset_index(drop=True)

    X_feature_cols = feature_cols + ['Time_Budget']
    X_train = train_df[X_feature_cols]
    X_test = test_df[X_feature_cols]

    # Deterministic tie-breaking for single-label targets using seed
    rng = np.random.RandomState(seed)
    y_train = np.array([rng.choice(w) for w in train_df['Tying_Winners']])
    y_test = np.array([rng.choice(w) for w in test_df['Tying_Winners']])

    # Multi-label / all-winners training set for Gradient Boosting Classifier
    train_all_df = train_df.explode('Tying_Winners').rename(columns={'Tying_Winners': 'Best_Algorithm'})
    X_train_all = train_all_df[X_feature_cols]
    y_train_all = train_all_df['Best_Algorithm']

    # Preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    reg_categorical_cols = categorical_cols + ['Algorithm']
    reg_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), reg_categorical_cols)
        ])

    # Compute Single Best Solver (SBS) strictly on training instances
    alg_cols = [f'Cost_{alg}' for alg in alg_names]
    inst_train_wins = {alg: 0.0 for alg in alg_names}
    for _, igroup in train_df.groupby('Instance'):
        time_wins = {alg: 0.0 for alg in alg_names}
        for _, row in igroup.iterrows():
            winners = row['Tying_Winners']
            for w in winners:
                time_wins[w] += 1.0 / len(winners)
        max_w = max(time_wins.values())
        top_algs = [a for a, w in time_wins.items() if np.isclose(w, max_w, rtol=1e-5)]
        for ta in top_algs:
            inst_train_wins[ta] += 1.0 / len(top_algs)

    sbs_alg = max(inst_train_wins, key=inst_train_wins.get)

    # Cost penalty computed from training data
    train_costs = train_df[alg_cols]
    max_train_cost = train_costs.replace(float('inf'), np.nan).max().max()
    max_cost_penalty = max_train_cost * 2 if pd.notna(max_train_cost) else 9999999

    bks_costs = test_df['BKS'].values
    sbs_costs = test_df[f'Cost_{sbs_alg}'].values
    sbs_costs_clean = np.where(sbs_costs == float('inf'), max_cost_penalty, sbs_costs)
    mean_sbs_cost = sbs_costs_clean.mean()
    sbs_bks_gap = np.mean(((sbs_costs_clean - bks_costs) / bks_costs) * 100)

    vbs_costs = test_df['Best_Cost'].values
    vbs_costs_clean = np.where(vbs_costs == float('inf'), max_cost_penalty, vbs_costs)
    mean_vbs_cost = vbs_costs_clean.mean()
    vbs_bks_gap = np.mean(((vbs_costs_clean - bks_costs) / bks_costs) * 100)
    vbs_gain_vs_sbs = ((mean_sbs_cost - mean_vbs_cost) / mean_sbs_cost) * 100

    def compute_metrics(y_pred, model_name):
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

        vbs_matches = sum(1 for i, pred in enumerate(y_pred) if pred in test_df.iloc[i]['Tying_Winners'])
        vbs_match_rate = (vbs_matches / len(test_df)) * 100

        aas_costs = np.array([test_df.iloc[i][f'Cost_{pred}'] for i, pred in enumerate(y_pred)])
        aas_costs = np.where(aas_costs == float('inf'), max_cost_penalty, aas_costs)
        mean_cost = aas_costs.mean()
        bks_gap = np.mean(((aas_costs - bks_costs) / bks_costs) * 100)
        gain_vs_sbs = ((mean_sbs_cost - mean_cost) / mean_sbs_cost) * 100

        return {
            'Seed': seed,
            'Model': model_name,
            'Exact Match Rate (%)': acc,
            'VBS Match Rate (%)': vbs_match_rate,
            'Precision (%)': prec,
            'Recall (%)': rec,
            'F1-Score (%)': f1,
            'Mean Solution Cost': mean_cost,
            'Gap to BKS (%)': bks_gap,
            'Cost Gain vs SBS (%)': gain_vs_sbs,
            'SBS Solver': abbreviate_alg(sbs_alg)
        }

    results = []

    # 1. Gradient Boosting Classifier
    gbc_model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, max_features=0.8, random_state=seed)
    gbc_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', gbc_model)])
    gbc_pipeline.fit(X_train_all, y_train_all)
    gbc_pred = gbc_pipeline.predict(X_test)
    results.append(compute_metrics(gbc_pred, 'Gradient Boosting Classifier'))

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=50, random_state=seed, n_jobs=-1)
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_model)])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    results.append(compute_metrics(rf_pred, 'Random Forest Classifier'))

    # 3. Multi-Layer Perceptron (MLP)
    mlp_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=250, activation='relu', alpha=0.01, random_state=seed)
    mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', mlp_model)])
    mlp_pipeline.fit(X_train, y_train)
    mlp_pred = mlp_pipeline.predict(X_test)
    results.append(compute_metrics(mlp_pred, 'Multi-Layer Perceptron (MLP)'))

    # 4. Gradient Boosting Regressor (Cost prediction)
    reg_train_mask = df_regression['Instance'].isin(train_instances)
    X_train_r = df_regression.loc[reg_train_mask, X_feature_cols + ['Algorithm']]
    y_train_r = df_regression.loc[reg_train_mask, 'Cost']

    gbr_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, max_features=0.8, random_state=seed)
    gbr_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', gbr_model)])
    gbr_pipeline.fit(X_train_r, y_train_r)

    # Batch test inference for regressor
    batch_features = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        feat = row[X_feature_cols].to_dict()
        for alg in alg_names:
            alg_feat = feat.copy()
            alg_feat['Algorithm'] = alg
            batch_features.append(alg_feat)

    batch_df = pd.DataFrame(batch_features)
    pred_costs = gbr_pipeline.predict(batch_df)
    pred_costs_matrix = pred_costs.reshape(len(test_df), len(alg_names))
    best_alg_indices = np.argmin(pred_costs_matrix, axis=1)
    gbr_pred = [alg_names[idx] for idx in best_alg_indices]
    results.append(compute_metrics(gbr_pred, 'Gradient Boosting Regressor'))

    # 5. LightGBM Ranker (Learning to Rank)
    from lightgbm import LGBMRanker
    train_reg_df = df_regression.loc[reg_train_mask].copy()
    train_reg_df.sort_values(by=['Instance', 'Time_Budget'], inplace=True)
    group_sizes = train_reg_df.groupby(['Instance', 'Time_Budget'], sort=False).size().values
    train_reg_df['Relevance'] = train_reg_df.groupby(['Instance', 'Time_Budget'], sort=False)['Cost'].rank(ascending=False, method='dense').astype(int)

    X_train_rank = train_reg_df[X_feature_cols + ['Algorithm']]
    y_train_rank = train_reg_df['Relevance']

    lgbm_ranker = LGBMRanker(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_samples=50, random_state=seed)
    ranker_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('ranker', lgbm_ranker)])
    ranker_pipeline.fit(X_train_rank, y_train_rank, ranker__group=group_sizes)

    pred_scores = ranker_pipeline.predict(batch_df)
    pred_scores_matrix = pred_scores.reshape(len(test_df), len(alg_names))
    best_rank_indices = np.argmax(pred_scores_matrix, axis=1)
    lgbm_pred = [alg_names[idx] for idx in best_rank_indices]
    results.append(compute_metrics(lgbm_pred, 'LightGBM Ranker (LTR)'))

    # 6. Single Best Solver (SBS) Baseline
    sbs_preds = [sbs_alg] * len(test_df)
    results.append(compute_metrics(sbs_preds, f'Single Best Solver (SBS: {abbreviate_alg(sbs_alg)})'))

    # 7. Uniform Random Selection Baseline
    rand_preds = rng.choice(alg_names, size=len(test_df))
    results.append(compute_metrics(rand_preds, 'Uniform Random Selection (1/9)'))

    # 8. Virtual Best Solver (VBS / Upper Bound)
    vbs_row = {
        'Seed': seed,
        'Model': 'Virtual Best Solver (VBS / Upper Bound)',
        'Exact Match Rate (%)': 100.0,
        'VBS Match Rate (%)': 100.0,
        'Precision (%)': 100.0,
        'Recall (%)': 100.0,
        'F1-Score (%)': 100.0,
        'Mean Solution Cost': mean_vbs_cost,
        'Gap to BKS (%)': vbs_bks_gap,
        'Cost Gain vs SBS (%)': vbs_gain_vs_sbs,
        'SBS Solver': abbreviate_alg(sbs_alg)
    }
    results.append(vbs_row)

    elapsed = time.time() - t_start
    print(f"Seed {seed:2d} completed in {elapsed:.1f}s | SBS: {abbreviate_alg(sbs_alg)} | Top Model Gain vs SBS: {max(r['Cost Gain vs SBS (%)'] for r in results if 'Classifier' in r['Model'] or 'Regressor' in r['Model'] or 'Ranker' in r['Model']):+.2f}%")
    return results


def run_multi_seed_benchmark(seeds=DEFAULT_SEEDS, data_path=DEFAULT_DATA_PATH, cache_path=DEFAULT_CACHE_PATH, output_dir='.'):
    """Runs the 10-seed experiment, aggregates results, and exports summary tables and charts."""
    print("=" * 95)
    print(f"ANYTIME ALGORITHM SELECTION BENCHMARK ACROSS {len(seeds)} SEEDS: {seeds}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 95)

    df_decisions, df_regression, alg_names, feature_cols, categorical_cols, numerical_cols = load_or_preprocess_data(data_path, cache_path)

    all_seed_results = []
    for idx, seed in enumerate(seeds, 1):
        print(f"\n--- Running Seed {seed} ({idx}/{len(seeds)}) ---")
        seed_res = evaluate_single_seed(seed, df_decisions, df_regression, alg_names, feature_cols, categorical_cols, numerical_cols)
        all_seed_results.extend(seed_res)

    results_df = pd.DataFrame(all_seed_results)

    # Standardize baseline model names across seeds for clean grouping
    def standardize_model_name(name):
        if name.startswith('Single Best Solver'):
            return 'Single Best Solver (SBS)'
        return name

    results_df['Strategy'] = results_df['Model'].apply(standardize_model_name)

    # Summary Statistics (Mean ± Std)
    metric_cols = ['Exact Match Rate (%)', 'VBS Match Rate (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'Mean Solution Cost', 'Gap to BKS (%)', 'Cost Gain vs SBS (%)']

    strategy_order = [
        'Gradient Boosting Classifier',
        'Random Forest Classifier',
        'Multi-Layer Perceptron (MLP)',
        'Gradient Boosting Regressor',
        'LightGBM Ranker (LTR)',
        'Single Best Solver (SBS)',
        'Uniform Random Selection (1/9)',
        'Virtual Best Solver (VBS / Upper Bound)'
    ]

    summary_rows = []
    for strat in strategy_order:
        sub = results_df[results_df['Strategy'] == strat]
        if len(sub) == 0:
            continue
        row = {'Selection Strategy / Model': strat}
        for m in metric_cols:
            mean_val = sub[m].mean()
            std_val = sub[m].std()
            if m == 'Mean Solution Cost':
                row[m] = f"{mean_val:,.2f} ± {std_val:.2f}"
            elif m == 'Cost Gain vs SBS (%)':
                row[m] = f"{mean_val:+.2f} ± {std_val:.2f}%"
            else:
                row[m] = f"{mean_val:.2f} ± {std_val:.2f}%"
            row[f'{m}_mean'] = mean_val
            row[f'{m}_std'] = std_val
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 105)
    print(f"TABLE: PERFORMANCE COMPARISON ACROSS {len(seeds)} RANDOM SEEDS (MEAN ± STD)")
    print("=" * 105)
    display_cols = ['Selection Strategy / Model', 'Exact Match Rate (%)', 'VBS Match Rate (%)', 'Mean Solution Cost', 'Gap to BKS (%)', 'Cost Gain vs SBS (%)']
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 105)

    # Export to Excel and CSV
    detailed_file = os.path.join(output_dir, 'anytime_selection_10_seeds_detailed.csv')
    summary_file = os.path.join(output_dir, 'anytime_selection_10_seeds_summary.xlsx')
    
    results_df.to_csv(detailed_file, index=False)
    with pd.ExcelWriter(summary_file) as writer:
        summary_df[display_cols].to_excel(writer, sheet_name='Summary (Mean ± Std)', index=False)
        summary_df.to_excel(writer, sheet_name='Summary All Metrics', index=False)
        results_df.to_excel(writer, sheet_name='Per-Seed Detailed Metrics', index=False)

    print(f"\nResults successfully exported:")
    print(f"  - Detailed CSV: {detailed_file}")
    print(f"  - Summary Excel: {summary_file}")

    # Generate multi-seed distribution plot
    plot_multi_seed_results(results_df, summary_df, output_dir)
    return results_df, summary_df


def plot_multi_seed_results(results_df, summary_df, output_dir='.'):
    """Generates multi-seed comparison plots (boxplots with jitter and error bar charts)."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    models_to_plot = [
        'Gradient Boosting Classifier',
        'Random Forest Classifier',
        'Multi-Layer Perceptron (MLP)',
        'Gradient Boosting Regressor',
        'LightGBM Ranker (LTR)'
    ]
    labels_clean = ['Gradient\nBoosting', 'Random\nForest', 'Multi-Layer\nPerceptron', 'GB\nRegressor', 'LightGBM\nRanker']

    plot_sub = results_df[results_df['Strategy'].isin(models_to_plot)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. VBS Match Rate Boxplot
    sns.boxplot(ax=axes[0], data=plot_sub, x='Strategy', y='VBS Match Rate (%)', order=models_to_plot, palette='Blues_r', width=0.5)
    sns.stripplot(ax=axes[0], data=plot_sub, x='Strategy', y='VBS Match Rate (%)', order=models_to_plot, color='black', size=5, jitter=0.2, alpha=0.7)
    axes[0].set_xticklabels(labels_clean, fontsize=10.5)
    axes[0].set_ylabel('VBS Match Rate (%) across 10 Seeds', fontsize=11.5, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Cost Gain vs SBS Boxplot
    sns.boxplot(ax=axes[1], data=plot_sub, x='Strategy', y='Cost Gain vs SBS (%)', order=models_to_plot, palette='Greens_r', width=0.5)
    sns.stripplot(ax=axes[1], data=plot_sub, x='Strategy', y='Cost Gain vs SBS (%)', order=models_to_plot, color='black', size=5, jitter=0.2, alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.2, label='SBS Baseline (0%)')
    axes[1].set_xticklabels(labels_clean, fontsize=10.5)
    axes[1].set_ylabel('Cost Gain vs. SBS (%) across 10 Seeds', fontsize=11.5, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].legend(loc='upper left', frameon=True)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_fig = os.path.join(output_dir, 'anytime_selection_10_seeds_boxplots.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-seed boxplot saved to: '{output_fig}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Anytime Algorithm Selection across multiple random seeds.')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS, help='List of random seeds (default: 1 2 3 4 5 6 7 8 9 10)')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH, help='Path to benchmark Excel file')
    parser.add_argument('--cache-path', type=str, default=DEFAULT_CACHE_PATH, help='Path to preprocessed pickle cache')
    parser.add_argument('--log-file', type=str, default='anytime_algorithm_selection_10_seeds.log', help='Log file path')
    parser.add_argument('--force-reload', action='store_true', help='Force reload and re-parse raw Excel instead of cache')

    args = parser.parse_args()

    logger = DualLogger(args.log_file)
    sys.stdout = logger

    try:
        run_multi_seed_benchmark(
            seeds=args.seeds,
            data_path=args.data_path,
            cache_path=args.cache_path
        )
    finally:
        sys.stdout = logger.terminal
        logger.close()
