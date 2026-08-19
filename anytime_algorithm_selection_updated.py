import os
import sys
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

# Global Constants
RANDOM_SEED = 2 #winner


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


class AnytimeAlgorithmSelectionUpdated:
    """
    Anytime Algorithm Selection (AAS) Framework with Instance-Wise 80/20 Train-Test Split,
    Single Best Solver (SBS) Baseline, Virtual Best Solver (VBS), and Benchmark Analytics.
    Standardized to Operations Research & Algorithm Selection literature terminology.
    """

    def __init__(self, data_path: str, random_seed: int = RANDOM_SEED, table_enabled: bool = True, plot_enabled: bool = True):
        self.data_path = data_path
        self.random_seed = random_seed
        self.table_enabled = table_enabled
        self.plot_enabled = plot_enabled
        self.df = None
        self.class_df = None
        self.reg_df = None
        self.feature_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate', 'Customers', 'Vehicles', 'Capacity']
        self.categorical_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate']
        self.numerical_cols = ['Customers', 'Vehicles', 'Capacity', 'Time_Budget']
        self.preprocessor = None
        self.sbs_alg = None  # Single Best Solver (SBS)
        self.mean_sbs_cost = None
        self.mean_vbs_cost = None  # Virtual Best Solver (VBS)
        self.max_cost_penalty = None
        self.alg_names = []
        self.train_instances = None
        self.test_instances = None
        self.best_pipeline = None
        self.best_y_pred = None
        
    def load_and_preprocess_data(self):
        """Loads dataset, resolves ties, and performs strictly instance-wise 80/20 split."""
        print("Loading benchmark data from Excel...")
        self.df = pd.read_excel(self.data_path)
        self.df['Algorithm'] = self.df['First Solution'] + '_' + self.df['Metaheuristic']
        self.alg_names = sorted(self.df['Algorithm'].unique().tolist())

        print("Preparing Anytime Algorithm Selection datasets across 1,000 instances...")
        instance_to_features = self.df.drop_duplicates(subset=['Instance']).set_index('Instance')[self.feature_cols]
        instance_to_bks = self.df.drop_duplicates(subset=['Instance']).set_index('Instance')['BKS Cost']
        time_steps = np.arange(0.5, 100.5, 0.5)

        classification_data = []
        regression_data = []

        grouped = self.df.groupby('Instance')
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
                
                seed_val = int(abs(hash(f"{instance}_{t}_{self.random_seed}")) % (2**31 - 1))
                rng = np.random.RandomState(seed_val)
                chosen_best_alg = rng.choice(tying_algs)

                # Regression rows
                for alg, cost in valid_costs.items():
                    reg_row = features.copy()
                    reg_row['Instance'] = instance
                    reg_row['Time_Budget'] = t
                    reg_row['Algorithm'] = alg
                    reg_row['Cost'] = cost
                    reg_row['BKS'] = bks
                    regression_data.append(reg_row)

                # Classification row
                class_row = features.copy()
                class_row['Instance'] = instance
                class_row['Time_Budget'] = t
                class_row['BKS'] = bks
                class_row['Best_Algorithm'] = chosen_best_alg
                class_row['Tying_Winners'] = tying_algs
                class_row['Num_Winners'] = len(tying_algs)
                class_row['Best_Cost'] = min_cost
                for alg in self.alg_names:
                    class_row[f'Cost_{alg}'] = alg_costs[alg].get(t, float('inf'))
                classification_data.append(class_row)

        self.class_df = pd.DataFrame(classification_data)
        self.reg_df = pd.DataFrame(regression_data)

        print(f"Classification dataset size: {len(self.class_df):,} decisions")
        print(f"Regression dataset size: {len(self.reg_df):,} samples")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])

        # Strictly Instance-Wise 80% Train / 20% Test Split
        alg_cols = [f'Cost_{alg}' for alg in self.alg_names]
        unique_instances = self.class_df['Instance'].unique()
        self.train_instances, self.test_instances = train_test_split(
            unique_instances, test_size=0.2, random_state=self.random_seed
        )

        train_mask = self.class_df['Instance'].isin(self.train_instances)
        test_mask = self.class_df['Instance'].isin(self.test_instances)

        X_class = self.class_df.drop(['Best_Algorithm', 'Tying_Winners', 'Num_Winners', 'Best_Cost', 'Instance', 'BKS'] + alg_cols, axis=1)
        self.X_class_columns = X_class.columns

        self.X_train = X_class[train_mask]
        self.X_test = X_class[test_mask]
        self.y_train = self.class_df.loc[train_mask, 'Best_Algorithm']
        self.y_test = self.class_df.loc[test_mask, 'Best_Algorithm']
        self.test_df = self.class_df[test_mask].reset_index(drop=True)

        # All-winners training set (multi-label / full credit for all tying winners)
        train_rows_all = []
        for _, row in self.class_df[train_mask].iterrows():
            for winner in row['Tying_Winners']:
                r = row[self.X_class_columns].to_dict()
                r['Best_Algorithm'] = winner
                train_rows_all.append(r)
        train_all_df = pd.DataFrame(train_rows_all)
        self.X_train_all = train_all_df[self.X_class_columns]
        self.y_train_all = train_all_df['Best_Algorithm']

        print(f"Unique instances: {len(unique_instances)} (Train: {len(self.train_instances)}, Test: {len(self.test_instances)})")
        print(f"Classification samples: Train = {len(self.X_train):,}, Test = {len(self.X_test):,}")

        # Instance-Wise Single Best Solver (SBS) on Training Instances
        inst_train_wins = {alg: 0.0 for alg in self.alg_names}
        for inst in self.train_instances:
            inst_data = self.class_df[self.class_df['Instance'] == inst]
            time_wins = {alg: 0.0 for alg in self.alg_names}
            for _, row in inst_data.iterrows():
                winners = row['Tying_Winners']
                for w in winners:
                    time_wins[w] += 1.0 / len(winners)
            max_w = max(time_wins.values())
            top_algs = [a for a, w in time_wins.items() if np.isclose(w, max_w, rtol=1e-5)]
            for ta in top_algs:
                inst_train_wins[ta] += 1.0 / len(top_algs)

        self.sbs_alg = max(inst_train_wins, key=inst_train_wins.get)
        print(f"\nSingle Best Solver (SBS on Training Set): {self.abbreviate_alg(self.sbs_alg)}")

        # Compute cost penalty from training data distribution
        train_costs = self.class_df.loc[train_mask, alg_cols]
        max_train_cost = train_costs.replace(float('inf'), np.nan).max().max()
        self.max_cost_penalty = max_train_cost * 2 if pd.notna(max_train_cost) else 9999999

        sbs_costs = self.test_df[f'Cost_{self.sbs_alg}']
        vbs_costs = self.test_df['Best_Cost']

        self.sbs_costs_clean = np.where(sbs_costs == float('inf'), self.max_cost_penalty, sbs_costs)
        self.vbs_costs_clean = np.where(vbs_costs == float('inf'), self.max_cost_penalty, vbs_costs)
        self.bks_costs = self.test_df['BKS'].values

        self.mean_sbs_cost = self.sbs_costs_clean.mean()
        self.mean_vbs_cost = self.vbs_costs_clean.mean()
        self.sbs_bks_gap = np.mean(((self.sbs_costs_clean - self.bks_costs) / self.bks_costs) * 100)
        self.vbs_bks_gap = np.mean(((self.vbs_costs_clean - self.bks_costs) / self.bks_costs) * 100)

    def evaluate_predictions(self, y_pred, model_name: str, pipeline=None):
        """Computes academic metrics including Exact Match Rate, VBS Match Rate, and BKS gap."""
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

        vbs_matches = sum(1 for i, pred in enumerate(y_pred) if pred in self.test_df.iloc[i]['Tying_Winners'])
        vbs_match_rate = (vbs_matches / len(self.test_df)) * 100

        aas_costs = np.array([self.test_df.iloc[i][f'Cost_{pred}'] for i, pred in enumerate(y_pred)])
        aas_costs = np.where(aas_costs == float('inf'), self.max_cost_penalty, aas_costs)
        mean_aas = aas_costs.mean()
        gain_vs_sbs = ((self.mean_sbs_cost - mean_aas) / self.mean_sbs_cost) * 100
        bks_gap = np.mean(((aas_costs - self.bks_costs) / self.bks_costs) * 100)

        print(f"--- {model_name} ---")
        print(f"  Exact Match Rate: {acc * 100:.2f}% | VBS Match Rate: {vbs_match_rate:.2f}%")
        print(f"  Mean Routing Cost: {mean_aas:.2f} (SBS: {self.mean_sbs_cost:.2f}, VBS: {self.mean_vbs_cost:.2f})")
        print(f"  Gap to BKS: {bks_gap:.2f}% (SBS Gap: {self.sbs_bks_gap:.2f}%)")
        print(f"  Cost Gain vs. SBS: {gain_vs_sbs:+.2f}%\n")

        return {
            'Model': model_name,
            'Exact Match Rate (%)': acc * 100,
            'VBS Match Rate (%)': vbs_match_rate,
            'Precision (%)': prec * 100,
            'Recall (%)': rec * 100,
            'F1-Score (%)': f1 * 100,
            'Mean Cost': mean_aas,
            'Gap to BKS (%)': bks_gap,
            'Cost Gain vs SBS (%)': gain_vs_sbs,
            'Pipeline': pipeline,
            'y_pred': y_pred
        }

    def train_random_forest(self):
        """Random Forest Classifier."""
        model = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=50, random_state=self.random_seed)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        return self.evaluate_predictions(y_pred, 'Random Forest Classifier', pipeline)

    def train_gradient_boosting_classifier(self):
        """Gradient Boosting Classifier trained on all winning labels."""
        model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, max_features=0.8, random_state=self.random_seed)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train_all, self.y_train_all)
        y_pred = pipeline.predict(self.X_test)
        self.best_pipeline = pipeline
        self.best_y_pred = y_pred
        return self.evaluate_predictions(y_pred, 'Gradient Boosting Classifier', pipeline)

    def train_lightgbm_ranker(self):
        """LightGBM Ranker (Learning-to-Rank) for algorithm selection."""
        from lightgbm import LGBMRanker

        # Ensure contiguous grouping for LightGBM
        train_reg_df = self.reg_df[self.reg_df['Instance'].isin(self.train_instances)].copy()
        train_reg_df.sort_values(by=['Instance', 'Time_Budget'], inplace=True)
        
        # Calculate group sizes (number of algorithms per instance-time pair)
        group_sizes = train_reg_df.groupby(['Instance', 'Time_Budget'], sort=False).size().values
        
        # Transform routing cost into a relevance score (higher integer = better performance)
        # Using dense ranking: tying minimal costs receive the identical highest relevance score.
        train_reg_df['Relevance'] = train_reg_df.groupby(['Instance', 'Time_Budget'], sort=False)['Cost'].rank(ascending=False, method='dense').astype(int)
        
        X_train_r = train_reg_df.drop(['Cost', 'Instance', 'BKS', 'Relevance'], axis=1)
        y_train_r = train_reg_df['Relevance']

        reg_categorical_cols = self.categorical_cols + ['Algorithm']
        reg_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), reg_categorical_cols)
            ])

        # Initialize Ranker (LambdaMART)
        ranker = LGBMRanker(n_estimators=250,learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_samples=50,random_state=self.random_seed)
        
        pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('ranker', ranker)])
        
        # Pass group parameter directly to the ranker via pipeline kwargs
        pipeline.fit(X_train_r, y_train_r, ranker__group=group_sizes)

        # Batch generation for inference matches the regressor logic
        batch_features = []
        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i]
            feat = row[self.X_class_columns].to_dict()
            for alg in self.alg_names:
                alg_feat = feat.copy()
                alg_feat['Algorithm'] = alg
                batch_features.append(alg_feat)

        batch_df = pd.DataFrame(batch_features)
        
        # Predict relevance scores
        pred_scores = pipeline.predict(batch_df)
        pred_scores_matrix = pred_scores.reshape(len(self.test_df), len(self.alg_names))
        
        # Ranker selects the algorithm with the highest predicted relevance
        best_alg_indices = np.argmax(pred_scores_matrix, axis=1)
        pred_algs = [self.alg_names[idx] for idx in best_alg_indices]

        return self.evaluate_predictions(pred_algs, 'LightGBM Ranker (LTR)', pipeline)

    def train_mlp_classifier(self):
        """Multi-Layer Perceptron Classifier."""
        model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=250, activation='relu', alpha=0.01, random_state=self.random_seed)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        return self.evaluate_predictions(y_pred, 'Multi-Layer Perceptron (MLP)', pipeline)

    def train_gradient_boosting_regressor(self):
        """Gradient Boosting Regressor (Cost prediction per algorithm)."""
        reg_train_mask = self.reg_df['Instance'].isin(self.train_instances)
        X_train_r = self.reg_df.loc[reg_train_mask].drop(['Cost', 'Instance', 'BKS'], axis=1)
        y_train_r = self.reg_df.loc[reg_train_mask, 'Cost']

        reg_categorical_cols = self.categorical_cols + ['Algorithm']
        reg_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), reg_categorical_cols)
            ])

        reg_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, max_features=0.8, random_state=self.random_seed)
        pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', reg_model)])
        pipeline.fit(X_train_r, y_train_r)

        batch_features = []
        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i]
            feat = row[self.X_class_columns].to_dict()
            for alg in self.alg_names:
                alg_feat = feat.copy()
                alg_feat['Algorithm'] = alg
                batch_features.append(alg_feat)

        batch_df = pd.DataFrame(batch_features)
        pred_costs = pipeline.predict(batch_df)
        pred_costs_matrix = pred_costs.reshape(len(self.test_df), len(self.alg_names))
        best_alg_indices = np.argmin(pred_costs_matrix, axis=1)
        pred_algs = [self.alg_names[idx] for idx in best_alg_indices]

        return self.evaluate_predictions(pred_algs, 'Gradient Boosting Regressor', pipeline)

    @staticmethod
    def abbreviate_alg(name: str) -> str:
        """Utility method to format verbose heuristic and metaheuristic names."""
        return (name.replace('Parallel Cheapest Insertion', 'PCI')
                    .replace('Guided Local Search (GLS)', 'GLS')
                    .replace('Tabu Search', 'TS')
                    .replace('Simulated Annealing', 'SA')
                    .replace('Savings (Clarke-Wright)', 'Savings')
                    .replace('_', ' + '))

    def display_algorithm_dominance_table(self):
        """Table 1: Empirical Instance Dominance across 1,000 instances."""
        if not self.table_enabled:
            return
        print("\n" + "=" * 95)
        print("TABLE 1: PERFORMANCE AND INSTANCE DOMINANCE OF CVRP ALGORITHM COMBINATIONS")
        print("         ACROSS 1,000 BENCHMARK INSTANCES")
        print("=" * 95)

        fractional_wins = {alg: 0.0 for alg in self.alg_names}
        instance_dom_counts = {alg: 0.0 for alg in self.alg_names}

        for instance, group in self.class_df.groupby('Instance'):
            inst_time_wins = {alg: 0.0 for alg in self.alg_names}
            for _, row in group.iterrows():
                winners = row['Tying_Winners']
                credit = 1.0 / len(winners)
                for w in winners:
                    fractional_wins[w] += credit
                    inst_time_wins[w] += credit
            max_w = max(inst_time_wins.values())
            top_algs = [a for a, w in inst_time_wins.items() if np.isclose(w, max_w, rtol=1e-5)]
            for ta in top_algs:
                instance_dom_counts[ta] += 1.0 / len(top_algs)

        sorted_algs = sorted(instance_dom_counts.items(), key=lambda x: x[1], reverse=True)
        alg_cols = [f'Cost_{alg}' for alg in self.alg_names]
        mean_costs = self.class_df[alg_cols].replace(float('inf'), np.nan).mean()
        bks_series = self.class_df['BKS'].values

        t1_rows = []
        for rank, (alg, inst_w) in enumerate(sorted_algs, 1):
            parts = alg.split('_')
            init_heur = self.abbreviate_alg(parts[0])
            meta_heur = self.abbreviate_alg(parts[1])
            m_cost = mean_costs[f'Cost_{alg}']
            pct_inst = (inst_w / 1000.0) * 100
            c_series = self.class_df[f'Cost_{alg}'].replace(float('inf'), np.nan).values
            m_gap = np.nanmean(((c_series - bks_series) / bks_series) * 100)

            t1_rows.append({
                'Rank': rank,
                'Initial Heuristic': init_heur,
                'Metaheuristic': meta_heur,
                'Dominant Instances': f"{inst_w:.1f}",
                'Instance Dominance (%)': f"{pct_inst:.2f}%",
                'Mean Solution Cost': f"{m_cost:,.2f}",
                'Mean Gap to BKS (%)': f"{m_gap:.2f}%"
            })
        df_t1 = pd.DataFrame(t1_rows)
        print(df_t1.to_string(index=False))

    def display_time_regime_table(self):
        """Table 2: Dominant Algorithm Combinations across Time-Budget Regimes."""
        if not self.table_enabled:
            return
        print("\n" + "=" * 90)
        print("TABLE 2: DOMINANT ALGORITHM COMBINATIONS (SBS IN REGIME) ACROSS TIME-BUDGET REGIMES")
        print("=" * 90)
        regimes = [
            ('Short (t <= 5s)', 0.0, 5.0),
            ('Medium (5s < t <= 20s)', 5.0, 20.0),
            ('Long (20s < t <= 50s)', 20.0, 50.0),
            ('Very Long (t > 50s)', 50.0, 100.0)
        ]
        t2_rows = []
        for reg_label, t_min, t_max in regimes:
            reg_sub = self.class_df[(self.class_df['Time_Budget'] > t_min) & (self.class_df['Time_Budget'] <= t_max)]
            n_inst = reg_sub['Instance'].nunique()

            inst_dom_wins = {alg: 0.0 for alg in self.alg_names}
            for inst, igroup in reg_sub.groupby('Instance'):
                i_wins = {alg: 0.0 for alg in self.alg_names}
                for _, row in igroup.iterrows():
                    winners = row['Tying_Winners']
                    for w in winners:
                        i_wins[w] += 1.0 / len(winners)
                max_w = max(i_wins.values())
                top_algs = [a for a, w in i_wins.items() if np.isclose(w, max_w, rtol=1e-5)]
                for ta in top_algs:
                    inst_dom_wins[ta] += 1.0 / len(top_algs)

            top_alg = max(inst_dom_wins, key=inst_dom_wins.get)
            top_inst_count = inst_dom_wins[top_alg]
            top_rate = (top_inst_count / n_inst) * 100 if n_inst > 0 else 0

            top_costs = reg_sub[f'Cost_{top_alg}'].replace(float('inf'), np.nan).values
            reg_bks = reg_sub['BKS'].values
            top_gap = np.nanmean(((top_costs - reg_bks) / reg_bks) * 100)

            t2_rows.append({
                'Time-Budget Regime': reg_label,
                'Evaluated Instances': n_inst,
                'SBS in Regime': self.abbreviate_alg(top_alg),
                'Instance Dominance (%)': f"{top_rate:.2f}%",
                'Dominant Gap to BKS (%)': f"{top_gap:.2f}%"
            })
        df_t2 = pd.DataFrame(t2_rows)
        print(df_t2.to_string(index=False))

    def display_feature_importance_table(self, rf_pipeline):
        """Table 3: Random Forest Gini Feature Importance."""
        if not self.table_enabled:
            return
        print("\n" + "=" * 90)
        print("TABLE 3: CVRP INSTANCE CHARACTERIZATION FEATURES AND RANDOM FOREST GINI FEATURE IMPORTANCE")
        print("=" * 90)
        cat_feature_names = list(rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(self.categorical_cols))
        all_feature_names = self.numerical_cols + cat_feature_names
        importances = rf_pipeline.named_steps['classifier'].feature_importances_

        feat_category_map = {
            'Customers': ('Problem Scale', 'Number of customer nodes ($N$)'),
            'Vehicles': ('Fleet Dimension', 'Available fleet capacity / number of vehicles ($K$)'),
            'Capacity': ('Capacity Constraint', 'Maximum vehicle payload capacity ($Q$)'),
            'Time_Budget': ('Operational Constraint', 'Elapsed execution runtime limit $t$ (seconds)'),
            'Depot Layout': ('Spatial Topology', 'Depot location geometry (Central, Eccentric, Random)'),
            'Cust Layout': ('Spatial Topology', 'Customer node dispersion (Random, Clustered, RC mixed)'),
            'Demand Type': ('Demand Profile', 'Customer demand distribution pattern'),
            'Route Class': ('Routing Geometry', 'Instance size and spatial clustering category'),
            'Climate': ('Environmental Factor', 'Topological impedance / weather condition category')
        }

        t3_rows = []
        for feat_name, imp_val in zip(all_feature_names, importances):
            base_name = feat_name
            for orig in self.feature_cols + ['Time_Budget']:
                if feat_name == orig or feat_name.startswith(orig + '_'):
                    base_name = orig
                    break
            cat, desc = feat_category_map.get(base_name, ('General', ''))
            t3_rows.append({
                'Feature Variable': feat_name,
                'Base Feature': base_name,
                'Category': cat,
                'Description': desc,
                'Gini Importance (%)': imp_val * 100
            })

        df_t3_raw = pd.DataFrame(t3_rows).sort_values(by='Gini Importance (%)', ascending=False)
        agg_feat_imp = df_t3_raw.groupby('Base Feature')['Gini Importance (%)'].sum().reset_index()
        agg_feat_imp['Category'] = agg_feat_imp['Base Feature'].apply(lambda x: feat_category_map.get(x, ('', ''))[0])
        agg_feat_imp['Description'] = agg_feat_imp['Base Feature'].apply(lambda x: feat_category_map.get(x, ('', ''))[1])
        agg_feat_imp = agg_feat_imp.sort_values(by='Gini Importance (%)', ascending=False)

        print("\n--- Aggregated Feature Importance by Characteristic Variable ---")
        print(agg_feat_imp[['Base Feature', 'Category', 'Gini Importance (%)', 'Description']].to_string(index=False))

    def display_selection_strategies_table(self, results):
        """Table 4: Selection Strategies Benchmark against SBS, Uniform Random, and VBS."""
        if not self.table_enabled:
            return
        print("\n" + "=" * 95)
        print("TABLE 4: AVERAGE SOLUTION COST AND PERCENTAGE GAP TO BEST KNOWN SOLUTION (BKS)")
        print("         ACROSS SELECTION STRATEGIES (200 UNSEEN TEST INSTANCES, 22,640 POINTS)")
        print("=" * 95)

        t4_rows = []
        for r in results:
            t4_rows.append({
                'Selection Strategy / Model': r['Model'],
                'Exact Match Rate (%)': f"{r['Exact Match Rate (%)']:.2f}%",
                'VBS Match Rate (%)': f"{r['VBS Match Rate (%)']:.2f}%",
                'Mean Solution Cost': f"{r['Mean Cost']:,.2f}",
                'Gap to BKS (%)': f"{r['Gap to BKS (%)']:.2f}%",
                'Gain vs. SBS (%)': f"{r['Cost Gain vs SBS (%)']:+.2f}%"
            })

        # Single Best Solver (SBS) Baseline
        sbs_preds = [self.sbs_alg] * len(self.y_test)
        sbs_acc = accuracy_score(self.y_test, sbs_preds) * 100
        sbs_opt = (sum(1 for i in range(len(self.test_df)) if self.sbs_alg in self.test_df.iloc[i]['Tying_Winners']) / len(self.test_df)) * 100
        t4_rows.append({
            'Selection Strategy / Model': f"Single Best Solver (SBS: {self.abbreviate_alg(self.sbs_alg)})",
            'Exact Match Rate (%)': f"{sbs_acc:.2f}%",
            'VBS Match Rate (%)': f"{sbs_opt:.2f}%",
            'Mean Solution Cost': f"{self.mean_sbs_cost:,.2f}",
            'Gap to BKS (%)': f"{self.sbs_bks_gap:.2f}%",
            'Gain vs. SBS (%)': "0.00%"
        })

        # Uniform Random Selection Baseline
        np.random.seed(self.random_seed)
        rand_preds = np.random.choice(self.alg_names, size=len(self.test_df))
        rand_costs = np.array([self.test_df.iloc[i][f'Cost_{rand_preds[i]}'] for i in range(len(rand_preds))])
        rand_costs = np.where(rand_costs == float('inf'), self.max_cost_penalty, rand_costs)
        rand_mean = rand_costs.mean()
        rand_bks_gap = np.mean(((rand_costs - self.bks_costs) / self.bks_costs) * 100)
        rand_gain = ((self.mean_sbs_cost - rand_mean) / self.mean_sbs_cost) * 100
        rand_acc = accuracy_score(self.y_test, rand_preds) * 100
        rand_opt = (sum(1 for i, pred in enumerate(rand_preds) if pred in self.test_df.iloc[i]['Tying_Winners']) / len(self.test_df)) * 100

        t4_rows.append({
            'Selection Strategy / Model': 'Uniform Random Selection (1/9)',
            'Exact Match Rate (%)': f"{rand_acc:.2f}%",
            'VBS Match Rate (%)': f"{rand_opt:.2f}%",
            'Mean Solution Cost': f"{rand_mean:,.2f}",
            'Gap to BKS (%)': f"{rand_bks_gap:.2f}%",
            'Gain vs. SBS (%)': f"{rand_gain:+.2f}%"
        })

        # Virtual Best Solver (VBS) Upper Bound
        vbs_gain = ((self.mean_sbs_cost - self.mean_vbs_cost) / self.mean_sbs_cost) * 100
        t4_rows.append({
            'Selection Strategy / Model': 'Virtual Best Solver (VBS / Upper Bound)',
            'Exact Match Rate (%)': "100.00%",
            'VBS Match Rate (%)': "100.00%",
            'Mean Solution Cost': f"{self.mean_vbs_cost:,.2f}",
            'Gap to BKS (%)': f"{self.vbs_bks_gap:.2f}%",
            'Gain vs. SBS (%)': f"{vbs_gain:+.2f}%"
        })

        df_t4 = pd.DataFrame(t4_rows)
        print(df_t4.to_string(index=False))

    def plot_optimality_gap_across_time(self, output_file='anytime_optimality_gap_bks_all_9_algorithms.png'):
        """Plot 1: Optimality gap across time budgets for all 9 algorithms (no top title)."""
        if not self.plot_enabled:
            return
        time_eval_points = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
        curve_results = {alg: [] for alg in self.alg_names}
        eval_t_axis = []

        for t_val in time_eval_points:
            sub = self.class_df[self.class_df['Time_Budget'] == t_val]
            if len(sub) == 0:
                continue
            eval_t_axis.append(t_val)
            sub_bks = sub['BKS'].values
            for alg in self.alg_names:
                c = sub[f'Cost_{alg}'].replace(float('inf'), np.nan).values
                g = np.nanmean(((c - sub_bks) / sub_bks) * 100)
                curve_results[alg].append(g)

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.figure(figsize=(11.5, 6.5), dpi=300)

        style_map = {
            'Parallel Cheapest Insertion_Guided Local Search (GLS)': ('#1f77b4', 'o', '-', 2.4, 'PCI + GLS (SBS)'),
            'Christofides_Guided Local Search (GLS)': ('#2ca02c', 's', '-', 2.2, 'Christofides + GLS'),
            'Savings (Clarke-Wright)_Guided Local Search (GLS)': ('#17becf', '^', '-', 1.8, 'Savings + GLS'),
            'Parallel Cheapest Insertion_Tabu Search': ('#ff7f0e', 'D', '--', 2.0, 'PCI + TS'),
            'Christofides_Tabu Search': ('#9467bd', 'v', '--', 1.8, 'Christofides + TS'),
            'Savings (Clarke-Wright)_Tabu Search': ('#d62728', '*', '--', 1.8, 'Savings + TS'),
            'Parallel Cheapest Insertion_Simulated Annealing': ('#8c564b', 'x', ':', 1.5, 'PCI + SA'),
            'Christofides_Simulated Annealing': ('#e377c2', 'P', ':', 1.5, 'Christofides + SA'),
            'Savings (Clarke-Wright)_Simulated Annealing': ('#7f7f7f', 'h', ':', 1.5, 'Savings + SA')
        }

        for alg in self.alg_names:
            color, marker, ls, lw, label = style_map.get(alg, ('#333333', 'o', '-', 1.5, self.abbreviate_alg(alg)))
            plt.plot(eval_t_axis, curve_results[alg], label=label, color=color, marker=marker,
                     linestyle=ls, linewidth=lw, markersize=6.5, alpha=0.92)

        plt.xlabel('Anytime Execution Budget $t$ (seconds)', fontsize=12, fontweight='bold', labelpad=8)
        plt.ylabel('Average Optimality Gap to BKS (%)', fontsize=12, fontweight='bold', labelpad=8)
        plt.xlim(0, 102)
        plt.xticks([0.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=10.5)
        plt.yticks(fontsize=10.5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=9.8, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, framealpha=0.95, edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Optimality gap plot saved to '{output_file}'.")

    def plot_selection_accuracy(self, results, output_file='anytime_selection_accuracy_comparison.png'):
        """Plot 2: Selection accuracy side-by-side (Exact Match vs VBS Match, no top title)."""
        if not self.plot_enabled:
            return
        plot_df = pd.DataFrame(results).set_index('Model')
        
        models_ordered = [
            'Gradient Boosting Classifier',
            'Random Forest Classifier',
            'Multi-Layer Perceptron (MLP)',
            'Gradient Boosting Regressor',
            'LightGBM Ranker (LTR)' if 'LightGBM Ranker (LTR)' in plot_df.index else 'LightGBM Ranker'
        ]
        
        labels_clean = [
            'Gradient\nBoosting',
            'Random\nForest',
            'Multi-Layer\nPerceptron',
            'GB\nRegressor',
            'LightGBM\nRanker'
        ]
        
        exact_accs = [plot_df.loc[m, 'Exact Match Rate (%)'] for m in models_ordered]
        vbs_match_accs = [plot_df.loc[m, 'VBS Match Rate (%)'] for m in models_ordered]

        plt.figure(figsize=(10.5, 5.8), dpi=300)
        x = np.arange(len(labels_clean))
        width = 0.35

        rects1 = plt.bar(x - width/2, exact_accs, width, label='Exact Match Rate (%)', color='#1f77b4', edgecolor='black', alpha=0.9)
        rects2 = plt.bar(x + width/2, vbs_match_accs, width, label='VBS Match Rate (%)', color='#2ca02c', edgecolor='black', alpha=0.9)

        plt.ylabel('Selection Match Rate (%)', fontsize=12, fontweight='bold', labelpad=8)
        plt.xticks(x, labels_clean, fontsize=10.5)
        plt.yticks(fontsize=10)
        plt.ylim(0, max(max(exact_accs), max(vbs_match_accs)) * 1.25)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend(loc='upper right', fontsize=10.5, framealpha=0.95, edgecolor='#cccccc')

        for rect in rects1:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#114b75')
        for rect in rects2:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., h + 0.5, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#166316')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Selection accuracy plot saved to '{output_file}'.")

    def plot_cost_gain_vs_static(self, results, output_file='anytime_cost_gain_vs_static.png'):
        """Plot 3: Cost Gain vs Single Best Solver (SBS) (no top title)."""
        if not self.plot_enabled:
            return
        plot_df = pd.DataFrame(results).set_index('Model')
        
        models_ordered = [
            'Gradient Boosting Classifier',
            'Random Forest Classifier',
            'Multi-Layer Perceptron (MLP)',
            'Gradient Boosting Regressor',
            'LightGBM Ranker (LTR)' if 'LightGBM Ranker (LTR)' in plot_df.index else 'LightGBM Ranker'
        ]
        
        labels_clean = [
            'Gradient\nBoosting',
            'Random\nForest',
            'Multi-Layer\nPerceptron',
            'GB\nRegressor',
            'LightGBM\nRanker'
        ]
        
        gains = [plot_df.loc[m, 'Cost Gain vs SBS (%)'] for m in models_ordered]

        plt.figure(figsize=(9.5, 5.5), dpi=300)
        x = np.arange(len(labels_clean))
        colors_gain = ['#1f77b4' if g >= 0 else '#d62728' for g in gains]
        rects_gain = plt.bar(x, gains, width=0.45, color=colors_gain, edgecolor='black', alpha=0.9)

        plt.axhline(0, color='black', linewidth=1.1, linestyle='-')
        plt.ylabel('Cost Gain vs. Single Best Solver (SBS) (%)', fontsize=12, fontweight='bold', labelpad=8)
        plt.xticks(x, labels_clean, fontsize=10.5)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        for rect in rects_gain:
            h = rect.get_height()
            va_pos = 'bottom' if h >= 0 else 'top'
            y_pos = h + 0.02 if h >= 0 else h - 0.05
            plt.text(rect.get_x() + rect.get_width()/2., y_pos, f'{h:+.2f}%', ha='center', va=va_pos, fontsize=10, fontweight='bold', color='#000000')

        y_min = min(gains) - 0.2 if min(gains) < 0 else -0.1
        y_max = max(gains) * 1.35 if max(gains) > 0 else 0.5
        plt.ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Cost gain plot saved to '{output_file}'.")

    def plot_confusion_matrix(self, output_file='anytime_aas_confusion_matrix.png'):
        """Plot 4: Normalized Confusion Matrix heatmap (no top title)."""
        if not self.plot_enabled or self.best_y_pred is None:
            return
        labels_short = [self.abbreviate_alg(a) for a in self.alg_names]
        cm_norm = confusion_matrix(self.y_test, self.best_y_pred, labels=self.alg_names, normalize='true') * 100

        plt.figure(figsize=(11, 9), dpi=300)
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=labels_short, yticklabels=labels_short,
                    cbar_kws={'label': 'Recall (% of True Instances)'})
        plt.xlabel('Predicted Solver', fontsize=11, fontweight='bold', labelpad=8)
        plt.ylabel('Actual Best Solver (Ground Truth)', fontsize=11, fontweight='bold', labelpad=8)
        plt.xticks(rotation=45, ha='right', fontsize=9.5)
        plt.yticks(rotation=0, fontsize=9.5)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix plot saved to '{output_file}'.\n")


if __name__ == '__main__':
    # Configuration flags & constant seed
    RANDOM_SEED = 68
    TABLE_ENABLED = True
    PLOT_ENABLED = True
    LOG_FILE = 'anytime_algorithm_selection.log'

    # Set up dual logging to both console and file
    logger = DualLogger(LOG_FILE)
    sys.stdout = logger

    try:
        start_time = datetime.now()
        print(f"{'=' * 95}")
        print(f"ANYTIME ALGORITHM SELECTION BENCHMARK RUN")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Logging output to: {LOG_FILE}")
        print(f"{'=' * 95}\n")

        data_file = 'or-tools_gaetano_benchmark_(1,000_random_instances)_20501208_174656.xlsx'
        aas = AnytimeAlgorithmSelectionUpdated(
            data_file,
            random_seed=RANDOM_SEED,
            table_enabled=TABLE_ENABLED,
            plot_enabled=PLOT_ENABLED
        )
        aas.load_and_preprocess_data()

        # Train all models
        results = [
            aas.train_gradient_boosting_classifier(),
            aas.train_random_forest(),
            aas.train_mlp_classifier(),
            aas.train_gradient_boosting_regressor(),
            aas.train_lightgbm_ranker()
        ]

        # Display tables (if table_enabled is True)
        aas.display_algorithm_dominance_table()
        aas.display_time_regime_table()
        rf_res = [r for r in results if 'Random Forest' in r['Model']][0]
        aas.display_feature_importance_table(rf_res['Pipeline'])
        aas.display_selection_strategies_table(results)

        # Generate plots (if plot_enabled is True)
        if PLOT_ENABLED:
            print("Generating updated figures...")
            aas.plot_optimality_gap_across_time('anytime_optimality_gap_bks_all_9_algorithms.png')
            aas.plot_selection_accuracy(results, 'anytime_selection_accuracy_comparison.png')
            aas.plot_cost_gain_vs_static(results, 'anytime_cost_gain_vs_static.png')
            aas.plot_confusion_matrix('anytime_aas_confusion_matrix.png')

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"\n{'=' * 95}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} (Elapsed: {elapsed})")
        print(f"All output successfully saved to: {LOG_FILE}")
        print(f"{'=' * 95}")

    finally:
        sys.stdout = logger.terminal
        logger.close()
