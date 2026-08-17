import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

class TwoStageAlgorithmSelection:
    """
    Two-Stage Framework for Anytime Algorithm Selection (AAS) in CVRP problems.
    Separates the selection into two distinct decision models:
      1. Phase 1 (Construction Heuristic Selector): f1(F, t) -> First Solution Strategy
      2. Phase 2 (Local Search Metaheuristic Selector): f2(F, t) -> Metaheuristic
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.class_df = None
        self.feature_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate', 'Customers', 'Vehicles', 'Capacity']
        self.categorical_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate']
        self.numerical_cols = ['Customers', 'Vehicles', 'Capacity', 'Time_Budget']
        self.preprocessor = None
        self.static_best_alg = None
        self.mean_static_cost = None
        self.mean_oracle_cost = None
        self.max_cost_penalty = None

    def load_and_preprocess_data(self):
        """Loads dataset and prepares separate target variables for Construction and Metaheuristic."""
        print("Loading data for Two-Stage Selection...")
        self.df = pd.read_excel(self.data_path)
        self.df['Algorithm'] = self.df['First Solution'] + '_' + self.df['Metaheuristic']

        print("Preparing dataset for Two-Stage Algorithm Selection...")
        instance_to_features = self.df.drop_duplicates(subset=['Instance']).set_index('Instance')[self.feature_cols]
        time_steps = np.arange(0.5, 100.5, 0.5)

        classification_data = []
        grouped = self.df.groupby('Instance')

        for instance, group in grouped:
            features = instance_to_features.loc[instance].to_dict()
            n_customers = features['Customers']
            max_t = min(0.5 * n_customers, 100.0)
            valid_ts = [t for t in time_steps if t <= max_t]

            alg_costs = {}
            for _, row in group.iterrows():
                alg = row['Algorithm']
                alg_costs[alg] = {}
                for t in valid_ts:
                    col_name = f'Best Cost @ {t:g}s'
                    if col_name in row and pd.notna(row[col_name]):
                        alg_costs[alg][t] = row[col_name]
                    else:
                        alg_costs[alg][t] = float('inf')

            for t in valid_ts:
                best_alg = None
                best_cost = float('inf')
                best_first_sol = None
                best_meta = None

                for _, row in group.iterrows():
                    alg = row['Algorithm']
                    cost = alg_costs[alg][t]
                    if cost < best_cost:
                        best_cost = cost
                        best_alg = alg
                        best_first_sol = row['First Solution']
                        best_meta = row['Metaheuristic']

                if best_alg is not None and best_cost != float('inf'):
                    class_row = features.copy()
                    class_row['Instance'] = instance
                    class_row['Time_Budget'] = t
                    class_row['Best_Algorithm'] = best_alg
                    class_row['Best_First_Solution'] = best_first_sol
                    class_row['Best_Metaheuristic'] = best_meta
                    class_row['Best_Cost'] = best_cost

                    for alg, costs in alg_costs.items():
                        class_row[f'Cost_{alg}'] = costs[t]
                    classification_data.append(class_row)

        self.class_df = pd.DataFrame(classification_data)
        print(f"Two-Stage dataset size: {len(self.class_df)}")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])

        alg_cols = [c for c in self.class_df.columns if c.startswith('Cost_')]
        X_class = self.class_df.drop(['Best_Algorithm', 'Best_First_Solution', 'Best_Metaheuristic', 'Best_Cost', 'Instance'] + alg_cols, axis=1)
        self.X_class_columns = X_class.columns

        y_first_sol = self.class_df['Best_First_Solution']
        y_meta = self.class_df['Best_Metaheuristic']
        indices = np.arange(len(self.class_df))

        self.X_train_idx, self.X_test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        self.X_train = X_class.iloc[self.X_train_idx]
        self.X_test = X_class.iloc[self.X_test_idx]

        self.y_train_first = y_first_sol.iloc[self.X_train_idx]
        self.y_test_first = y_first_sol.iloc[self.X_test_idx]

        self.y_train_meta = y_meta.iloc[self.X_train_idx]
        self.y_test_meta = y_meta.iloc[self.X_test_idx]

        # Calculate static best solver baseline
        train_costs = self.class_df.iloc[self.X_train_idx][alg_cols]
        self.static_best_alg = train_costs.mean().idxmin().replace('Cost_', '')
        print(f"Static Best Overall Solver: {self.static_best_alg}\n")

        self.test_df = self.class_df.iloc[self.X_test_idx]
        static_costs = self.test_df[f'Cost_{self.static_best_alg}']
        oracle_costs = self.test_df['Best_Cost']

        self.max_cost_penalty = self.test_df[alg_cols].replace(float('inf'), np.nan).max().max() * 2
        if pd.isna(self.max_cost_penalty):
            self.max_cost_penalty = 9999999

        static_costs_clean = np.where(static_costs == float('inf'), self.max_cost_penalty, static_costs)
        oracle_costs_clean = np.where(oracle_costs == float('inf'), self.max_cost_penalty, oracle_costs)
        self.mean_static_cost = static_costs_clean.mean()
        self.mean_oracle_cost = oracle_costs_clean.mean()

    def evaluate_two_stage_models(self, model_name: str, model_first, model_meta):
        """Train two separate models (Construction & Metaheuristic) and evaluate accuracy & cost gain."""
        print(f"--- Two-Stage Selection with {model_name} ---")
        
        # Pipeline 1: Construction Heuristic Predictor
        pipe_first = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model_first)])
        pipe_first.fit(self.X_train, self.y_train_first)
        pred_first = pipe_first.predict(self.X_test)
        acc_first = accuracy_score(self.y_test_first, pred_first)

        # Pipeline 2: Metaheuristic Predictor
        pipe_meta = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model_meta)])
        pipe_meta.fit(self.X_train, self.y_train_meta)
        pred_meta = pipe_meta.predict(self.X_test)
        acc_meta = accuracy_score(self.y_test_meta, pred_meta)

        # Combined Prediction (First_Solution + '_' + Metaheuristic)
        combined_preds = [f"{f}_{m}" for f, m in zip(pred_first, pred_meta)]

        # Lookup combined cost
        aas_costs = np.array([self.test_df.iloc[i].get(f'Cost_{comb}', self.max_cost_penalty) for i, comb in enumerate(combined_preds)])
        aas_costs = np.where(aas_costs == float('inf'), self.max_cost_penalty, aas_costs)
        mean_aas = aas_costs.mean()
        gain = ((self.mean_static_cost - mean_aas) / self.mean_static_cost) * 100

        print(f"  Phase 1 (Construction Heuristic) Accuracy: {acc_first:.4f}")
        print(f"  Phase 2 (Metaheuristic) Accuracy:           {acc_meta:.4f}")
        print(f"  Two-Stage Combined Mean Routing Cost:       {mean_aas:.2f} (Static: {self.mean_static_cost:.2f}, Oracle: {self.mean_oracle_cost:.2f})")
        print(f"  Improvement over Static Best:               {gain:.2f}%\n")

        # Feature Importance Analysis for Random Forest
        if model_name == "Random Forest":
            feature_names = self.numerical_cols + list(pipe_first.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(self.categorical_cols))
            
            imp_first = pd.Series(pipe_first.named_steps['classifier'].feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
            imp_meta = pd.Series(pipe_meta.named_steps['classifier'].feature_importances_, index=feature_names).sort_values(ascending=False).head(5)
            
            print("  Top 5 Features driving Construction Heuristic Selection:")
            for feat, val in imp_first.items():
                print(f"    - {feat}: {val:.4f}")
            print("\n  Top 5 Features driving Metaheuristic Selection:")
            for feat, val in imp_meta.items():
                print(f"    - {feat}: {val:.4f}")
            print("\n" + "="*60 + "\n")

    def run_all(self):
        """Executes Two-Stage evaluation across Random Forest, Gradient Boosting, and MLP."""
        self.evaluate_two_stage_models(
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        )

        self.evaluate_two_stage_models(
            "Multi-Layer Perceptron (MLP)",
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        )

        self.evaluate_two_stage_models(
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=50, random_state=42),
            GradientBoostingClassifier(n_estimators=50, random_state=42)
        )


if __name__ == '__main__':
    data_file = 'or-tools_gaetano_benchmark_(1,000_random_instances)_20501208_174656.xlsx'
    two_stage = TwoStageAlgorithmSelection(data_file)
    two_stage.load_and_preprocess_data()
    two_stage.run_all()
