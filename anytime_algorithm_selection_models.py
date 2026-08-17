import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

class AnytimeAlgorithmSelection:
    """
    Framework for Anytime Algorithm Selection (AAS) in CVRP problems.
    Supports Random Forest, Gradient Boosting, Multi-Layer Perceptron (Classifiers),
    and Gradient Boosting Regressor (Cost prediction approach).
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.class_df = None
        self.reg_df = None
        self.feature_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate', 'Customers', 'Vehicles', 'Capacity']
        self.categorical_cols = ['Depot Layout', 'Cust Layout', 'Demand Type', 'Route Class', 'Climate']
        self.numerical_cols = ['Customers', 'Vehicles', 'Capacity', 'Time_Budget']
        self.preprocessor = None
        self.static_best_alg = None
        self.mean_static_cost = None
        self.mean_oracle_cost = None
        self.max_cost_penalty = None
        
    def load_and_preprocess_data(self):
        """Loads dataset and prepares classification and regression datasets."""
        print("Loading data...")
        self.df = pd.read_excel(self.data_path)
        self.df['Algorithm'] = self.df['First Solution'] + '_' + self.df['Metaheuristic']

        print("Preparing dataset for Algorithm Selection...")
        instance_to_features = self.df.drop_duplicates(subset=['Instance']).set_index('Instance')[self.feature_cols]
        time_steps = np.arange(0.5, 100.5, 0.5)

        classification_data = []
        regression_data = []

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

                for alg, costs in alg_costs.items():
                    cost = costs[t]
                    if cost != float('inf'):
                        reg_row = features.copy()
                        reg_row['Time_Budget'] = t
                        reg_row['Algorithm'] = alg
                        reg_row['Cost'] = cost
                        regression_data.append(reg_row)

                    if cost < best_cost:
                        best_cost = cost
                        best_alg = alg

                if best_alg is not None and best_cost != float('inf'):
                    class_row = features.copy()
                    class_row['Instance'] = instance
                    class_row['Time_Budget'] = t
                    class_row['Best_Algorithm'] = best_alg
                    class_row['Best_Cost'] = best_cost
                    for alg, costs in alg_costs.items():
                        class_row[f'Cost_{alg}'] = costs[t]
                    classification_data.append(class_row)

        self.class_df = pd.DataFrame(classification_data)
        self.reg_df = pd.DataFrame(regression_data)

        print(f"Classification dataset size: {len(self.class_df)}")
        print(f"Regression dataset size: {len(self.reg_df)}")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ])

        # Prepare train/test splits
        alg_cols = [c for c in self.class_df.columns if c.startswith('Cost_')]
        X_class = self.class_df.drop(['Best_Algorithm', 'Best_Cost', 'Instance'] + alg_cols, axis=1)
        self.X_class_columns = X_class.columns
        y_class = self.class_df['Best_Algorithm']
        indices = np.arange(len(self.class_df))

        self.X_train_idx, self.X_test_idx, self.y_train, self.y_test = train_test_split(
            indices, y_class, test_size=0.2, random_state=42
        )
        self.X_train = X_class.iloc[self.X_train_idx]
        self.X_test = X_class.iloc[self.X_test_idx]

        # Calculate static best solver
        train_costs = self.class_df.iloc[self.X_train_idx][alg_cols]
        self.static_best_alg = train_costs.mean().idxmin().replace('Cost_', '')
        print(f"\nStatic Best Overall Solver: {self.static_best_alg}")

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

    def train_random_forest(self):
        """Method for Random Forest Classification Approach."""
        print("--- Training Random Forest Classifier ---")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train, self.y_train)

        y_pred = pipeline.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        
        aas_costs = np.array([self.test_df.iloc[i][f'Cost_{pred}'] for i, pred in enumerate(y_pred)])
        aas_costs = np.where(aas_costs == float('inf'), self.max_cost_penalty, aas_costs)
        mean_aas = aas_costs.mean()
        gain = ((self.mean_static_cost - mean_aas) / self.mean_static_cost) * 100

        print(f"Accuracy: {acc:.4f}")
        print(f"Mean Cost: {mean_aas:.2f} (Static: {self.mean_static_cost:.2f}, Oracle: {self.mean_oracle_cost:.2f})")
        print(f"Improvement over Static: {gain:.2f}%\n")

        # Feature Importance Analysis
        feature_names = self.numerical_cols + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(self.categorical_cols))
        importances = pd.Series(pipeline.named_steps['classifier'].feature_importances_, index=feature_names).sort_values(ascending=False)
        print("  Top Features driving Algorithm Selection (Joint Model):")
        for feat, val in importances.head(8).items():
            print(f"    - {feat}: {val:.4f}")
        print("\n")
        return pipeline

    def train_gradient_boosting_classifier(self):
        """Method for Gradient Boosting Classification Approach."""
        print("--- Training Gradient Boosting Classifier ---")
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train, self.y_train)

        y_pred = pipeline.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)

        aas_costs = np.array([self.test_df.iloc[i][f'Cost_{pred}'] for i, pred in enumerate(y_pred)])
        aas_costs = np.where(aas_costs == float('inf'), self.max_cost_penalty, aas_costs)
        mean_aas = aas_costs.mean()
        gain = ((self.mean_static_cost - mean_aas) / self.mean_static_cost) * 100

        print(f"Accuracy: {acc:.4f}")
        print(f"Mean Cost: {mean_aas:.2f} (Static: {self.mean_static_cost:.2f}, Oracle: {self.mean_oracle_cost:.2f})")
        print(f"Improvement over Static: {gain:.2f}%\n")
        return pipeline

    def train_mlp_classifier(self):
        """Method for Multi-Layer Perceptron Classification Approach."""
        print("--- Training Multi-Layer Perceptron Classifier ---")
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', model)])
        pipeline.fit(self.X_train, self.y_train)

        y_pred = pipeline.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)

        aas_costs = np.array([self.test_df.iloc[i][f'Cost_{pred}'] for i, pred in enumerate(y_pred)])
        aas_costs = np.where(aas_costs == float('inf'), self.max_cost_penalty, aas_costs)
        mean_aas = aas_costs.mean()
        gain = ((self.mean_static_cost - mean_aas) / self.mean_static_cost) * 100

        print(f"Accuracy: {acc:.4f}")
        print(f"Mean Cost: {mean_aas:.2f} (Static: {self.mean_static_cost:.2f}, Oracle: {self.mean_oracle_cost:.2f})")
        print(f"Improvement over Static: {gain:.2f}%\n")
        return pipeline

    def train_gradient_boosting_regressor(self):
        """Method for Gradient Boosting Regressor Approach (Cost Prediction)."""
        print("--- Training Gradient Boosting Regressor ---")
        X_reg = self.reg_df.drop('Cost', axis=1)
        y_reg = self.reg_df['Cost']

        reg_categorical_cols = self.categorical_cols + ['Algorithm']
        reg_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), reg_categorical_cols)
            ])

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', reg_model)])
        pipeline.fit(X_train_r, y_train_r)

        # Full evaluation on full test set
        alg_cols = [c for c in self.class_df.columns if c.startswith('Cost_')]
        algs = [c.replace('Cost_', '') for c in alg_cols]
        batch_features = []

        X_class_cols = self.X_class_columns
        for i in range(len(self.test_df)):
            row = self.test_df.iloc[i]
            feat = row[X_class_cols].to_dict()
            for alg in algs:
                alg_feat = feat.copy()
                alg_feat['Algorithm'] = alg
                batch_features.append(alg_feat)

        batch_df = pd.DataFrame(batch_features)
        pred_costs = pipeline.predict(batch_df)

        pred_costs_matrix = pred_costs.reshape(len(self.test_df), len(algs))
        best_alg_indices = np.argmin(pred_costs_matrix, axis=1)

        reg_aas_costs = np.array([self.test_df.iloc[i][f'Cost_{algs[idx]}'] for i, idx in enumerate(best_alg_indices)])
        reg_aas_costs = np.where(reg_aas_costs == float('inf'), self.max_cost_penalty, reg_aas_costs)

        mean_reg_aas = reg_aas_costs.mean()
        gain = ((self.mean_static_cost - mean_reg_aas) / self.mean_static_cost) * 100

        print(f"Mean Cost: {mean_reg_aas:.2f} (Static: {self.mean_static_cost:.2f}, Oracle: {self.mean_oracle_cost:.2f})")
        print(f"Improvement over Static: {gain:.2f}%\n")
        return pipeline


if __name__ == '__main__':
    data_file = 'or-tools_gaetano_benchmark_(1,000_random_instances)_20501208_174656.xlsx'
    aas = AnytimeAlgorithmSelection(data_file)
    aas.load_and_preprocess_data()

    # One method per each approach / model
    aas.train_random_forest()
    aas.train_gradient_boosting_classifier()
    aas.train_mlp_classifier()
    aas.train_gradient_boosting_regressor()
