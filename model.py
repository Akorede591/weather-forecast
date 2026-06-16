import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# 1. SYSTEM CONFIGURATION & COMPONENT SETUP
# =====================================================================
# Absolute file path to your dataset
FILE_PATH = r"C:\Users\USER\Desktop\Weather Forecasting Prediction System\weather_prediction_dataset2.csv"

# The column name from your CSV that you want to forecast (e.g., 'Temperature', 'Rainfall')
TARGET_COLUMN = "Target_Next_Temp"  


# =====================================================================
# 2. BINARY PSO FEATURE SELECTION CLASS
# =====================================================================
class BPSOFeatureSelection:
    """
    Implements Binary Particle Swarm Optimization for selecting the most
    predictive weather feature subset while stripping out redundant noise.
    """
    def __init__(self, X, y, num_particles=12, max_iter=15, alpha=0.85):
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha  # Weight balancing accuracy (1.0) against dimensionality reduction (0.0)
        self.num_features = X.shape[1]
        
        # Internal validation split to score the fitness of each particle configuration
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
    def _sigmoid(self, x):
        """Maps continuous velocity boundaries into binary selection probabilities."""
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _calculate_fitness(self, position):
        """Trains a lightweight evaluator network to calculate subset error."""
        selected_indices = np.where(position == 1)[0]
        
        # Penalty rule if the particle switches off all features
        if len(selected_indices) == 0:
            return float('inf')
        
        X_tr_sub = self.X_train[:, selected_indices]
        X_val_sub = self.X_val[:, selected_indices]
        
        # Fast-converging estimator BPNN used for scoring subsets
        evaluator = MLPRegressor(
            hidden_layer_sizes=(12,), 
            max_iter=100, 
            random_state=42,
            early_stopping=True
        )
        
        try:
            evaluator.fit(X_tr_sub, self.y_train)
            predictions = evaluator.predict(X_val_sub)
            mse = mean_squared_error(self.y_val, predictions)
        except Exception:
            mse = float('inf')
            
        # Fitness objective function math: minimize error and feature count
        feature_ratio = len(selected_indices) / self.num_features
        fitness_score = (self.alpha * mse) + ((1 - self.alpha) * feature_ratio)
        return fitness_score

    def optimize(self):
        """Executes the discrete swarm optimization loop."""
        # Initialize velocity vectors and discrete binary positions
        velocities = np.random.uniform(-4, 4, (self.num_particles, self.num_features))
        positions = np.random.randint(2, size=(self.num_particles, self.num_features))
        
        p_best_positions = np.copy(positions)
        p_best_scores = np.array([self._calculate_fitness(p) for p in positions])
        
        g_best_idx = np.argmin(p_best_scores)
        g_best_position = np.copy(p_best_positions[g_best_idx])
        g_best_score = p_best_scores[g_best_idx]
        
        # Swarm acceleration parameters
        w, c1, c2 = 0.7, 1.5, 1.5
        
        print("\n--- Starting BPSO Swarm Feature Optimization ---")
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.num_features), np.random.rand(self.num_features)
                
                # Update velocity vectors
                cognitive = c1 * r1 * (p_best_positions[i] - positions[i])
                social = c2 * r2 * (g_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Apply velocity thresholding
                velocities[i] = np.clip(velocities[i], -4, 4)
                
                # Probability mapping to flip bits (select/deselect)
                probs = self._sigmoid(velocities[i])
                positions[i] = (np.random.rand(self.num_features) < probs).astype(int)
                
                # Evaluate new positions
                current_score = self._calculate_fitness(positions[i])
                
                # Update personal records
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = np.copy(positions[i])
                    
            # Update swarm historical record holder
            best_iter_idx = np.argmin(p_best_scores)
            if p_best_scores[best_iter_idx] < g_best_score:
                g_best_score = p_best_scores[best_iter_idx]
                g_best_position = np.copy(p_best_positions[best_iter_idx])
                
            print(f" Iteration {iteration+1:02d}/{self.max_iter} | Best Global Fitness Cost: {g_best_score:.6f}")
            
        return g_best_position


# =====================================================================
# 3. PIPELINE DATA LOADER & SYSTEM RUNTIME
# =====================================================================
if __name__ == "__main__":
    print(f"Loading dataset from: {FILE_PATH}...")
    try:
        raw_data = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"\n[Error]: Could not locate dataset at '{FILE_PATH}'. Check file path syntax.")
        exit()

    # --- DYNAMIC TARGET INPUT VALIDATION ENGINE ---
    if TARGET_COLUMN not in raw_data.columns:
        print("\n" + "!"*65)
        print(f"CRITICAL FAULT: '{TARGET_COLUMN}' is not a column in your CSV file.")
        print("Please review the available column names detected in your file below:")
        print("-" * 65)
        for col in raw_data.columns:
            print(f" -> {col}")
        print("-" * 65)
        print("REMEDY: Copy one of the names above and paste it into line 14 of this code.")
        print("!"*65 + "\n")
        exit()

    # Strip rows where target target metrics are missing
    raw_data = raw_data.dropna(subset=[TARGET_COLUMN])
    
    # Isolate targets and structural input features
    feature_columns = [col for col in raw_data.columns if col != TARGET_COLUMN]
    X_raw = raw_data[feature_columns].values
    y_raw = raw_data[TARGET_COLUMN].values
    
    # Impute missing values (NaNs) across features using feature column averages
    if np.isnan(X_raw).any():
        print("Notice: Missing entries detected in feature matrix. Applying mean imputation...")
        col_means = np.nanmean(X_raw, axis=0)
        inds = np.where(np.isnan(X_raw))
        X_raw[inds] = np.take(col_means, inds[1])

    # Scaler transformation mapping data range constraints to bounds [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 1. Execute optimization framework to determine optimal inputs
    pso_engine = BPSOFeatureSelection(X_scaled, y_raw, num_particles=12, max_iter=15, alpha=0.85)
    optimal_mask = pso_engine.optimize()
    
    selected_features = [feature_columns[i] for i, mask in enumerate(optimal_mask) if mask == 1]
    
    print("\n" + "="*60)
    print("PSO OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Raw Input Feature Space Dimension Count: {X_raw.shape[1]}")
    print(f"Optimized Feature Subset Selected:       {selected_features}")
    print("="*60 + "\n")
    
    # Filter matrix layers to contain only the optimized feature indices
    X_optimized = X_scaled[:, np.where(optimal_mask == 1)[0]]
    
    # Final data validation segment partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X_optimized, y_raw, test_size=0.2, random_state=42
    )
    
    # 2. Design and compile the Backpropagation Forecasting Architecture (BPNN)
    print("Training structural Backpropagation Neural Network (BPNN)...")
    forecasting_bpnn = MLPRegressor(
        hidden_layer_sizes=(32, 16), # Dense multi-tier hidden topography
        activation='relu',           # Non-linear thresholding function
        solver='adam',               # Backprop gradient optimizer
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=True          # Prevents model overfitting
    )
    
    forecasting_bpnn.fit(X_train, y_train)
    predictions = forecasting_bpnn.predict(X_test)
    
    # 3. System Metrics Evaluation Runtime
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*60)
    print("FINAL SYSTEM FORECAST PERFORMANCE EVALUATION REPORT")
    print("="*60)
    print(f"Mean Absolute Error (MAE):         {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE):     {rmse:.4f}")
    print(f"Coefficient of Determination (R²):  {r2:.4f}")
    print("="*60 + "\n")