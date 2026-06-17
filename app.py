import numpy as np
import pandas as pd
import streamlit as st
import graphviz
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="PSO-BPNN Weather Predictor", layout="wide")

# ==========================================
# DIAGRAM GENERATION FUNCTIONS (ENHANCED SIZE)
# ==========================================
def generate_pipeline_flowchart():
    """Generates a large-scale main system architecture pipeline chart."""
    dot = graphviz.Digraph(comment='System Architecture')
    dot.attr(rankdir='LR', size='16,8!', ratio='fill', bgcolor='transparent')
    
    dot.attr('node', shape='box', style='filled,rounded', color='#1E88E5', 
             fontcolor='white', fontname='Helvetica-Bold', fontsize='14', height='1.2', width='2.5')
    
    dot.node('A', 'User CSV Upload\n(Dynamic Ingestion)')
    dot.node('B', 'Data Preprocessing\n(MinMax Scaling & Lags)')
    
    dot.attr('node', color='#43A047')
    dot.node('C', 'Binary Particle Swarm\nOptimization (BPSO)')
    dot.node('D', 'Optimal Feature\nSubset Matrix')
    
    dot.attr('node', color='#E53935')
    dot.node('E', 'Backpropagation Neural\nNetwork (BPNN)')
    dot.node('F', 'Weather Forecast\nOutput Report')
    
    dot.attr('edge', fontname='Helvetica', fontsize='11', penwidth='2.0', color='#424242')
    dot.edge('A', 'B')
    dot.edge('B', 'C', label=' All Features Space')
    dot.edge('B', 'D', style='dashed', label=' Applies Filter')
    dot.edge('C', 'D', label=' Generates Mask')
    dot.edge('D', 'E', label=' Clean Input Matrix')
    dot.edge('E', 'F')
    
    return dot

def generate_algo_flowchart(selected_feats):
    """Generates a high-resolution workflow diagram of the evolutionary BPSO loop."""
    dot = graphviz.Digraph(comment='Algorithm Decision Loop')
    dot.attr(rankdir='TB', size='14,12!', ratio='fill', bgcolor='transparent')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', height='1.0', width='2.8')
    dot.attr('edge', penwidth='1.8', color='#616161')
    
    dot.node('Start', 'Initialize Swarm Particles\n(Random Binary Masks)', shape='ellipse', color='#757575', fontcolor='white')
    dot.node('Fit', 'Evaluate Fitness\nScore = f(MSE, Feature Count)', shape='box', color='#1E88E5', fontcolor='white')
    dot.node('Update', 'Update Particle Velocities & Positions\n(Sigmoid Probability Mapping)', shape='box', color='#1E88E5', fontcolor='white')
    dot.node('Check', 'Max Iterations Reached?', shape='diamond', color='#FDD835', fontcolor='black', height='1.5')
    
    features_str = "\\n".join(selected_feats[:6]) + ("\\n..." if len(selected_feats) > 6 else "")
    dot.node('End', f'BPNN Inputs Assigned:\\n{features_str}', shape='parallelogram', color='#43A047', fontcolor='white', width='3.5')
    
    dot.edge('Start', 'Fit')
    dot.edge('Fit', 'Update')
    dot.edge('Update', 'Check')
    dot.edge('Check', 'Fit', label=' No (Next Gen)')
    dot.edge('Check', 'End', label=' Yes (Optimum Subset)')
    
    return dot

# ==========================================
# HIGH-SPEED BINARY PSO ENGINE
# ==========================================
class BPSOFeatureSelection:
    def __init__(self, X, y, num_particles, max_iter, alpha=0.85):
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.num_features = X.shape[1]
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def _calculate_fitness(self, position):
        selected_indices = np.where(position == 1)[0]
        if len(selected_indices) == 0:
            return float('inf')
        
        X_tr_sub = self.X_train[:, selected_indices]
        X_val_sub = self.X_val[:, selected_indices]
        
        evaluator = SGDRegressor(max_iter=20, random_state=42, tol=1e-3)
        try:
            evaluator.fit(X_tr_sub, self.y_train)
            mse = mean_squared_error(self.y_val, evaluator.predict(X_val_sub))
        except:
            mse = float('inf')
            
        feature_ratio = len(selected_indices) / self.num_features
        return (self.alpha * mse) + ((1 - self.alpha) * feature_ratio)

    def optimize(self, progress_bar, status_text):
        velocities = np.random.uniform(-4, 4, (self.num_particles, self.num_features))
        positions = np.random.randint(2, size=(self.num_particles, self.num_features))
        
        p_best_positions = np.copy(positions)
        p_best_scores = np.array([self._calculate_fitness(p) for p in positions])
        
        g_best_idx = np.argmin(p_best_scores)
        g_best_position = np.copy(p_best_positions[g_best_idx])
        g_best_score = p_best_scores[g_best_idx]
        
        w, c1, c2 = 0.7, 1.5, 1.5
        
        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.num_particles, self.num_features)
            r2 = np.random.rand(self.num_particles, self.num_features)
            
            cognitive = c1 * r1 * (p_best_positions - positions)
            social = c2 * r2 * (g_best_position - positions)
            velocities = w * velocities + cognitive + social
            velocities = np.clip(velocities, -4, 4)
            
            positions = (np.random.rand(self.num_particles, self.num_features) < self._sigmoid(velocities)).astype(int)
            
            for i in range(self.num_particles):
                current_score = self._calculate_fitness(positions[i])
                if current_score < p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_positions[i] = np.copy(positions[i])
                    
            best_iter_idx = np.argmin(p_best_scores)
            if p_best_scores[best_iter_idx] < g_best_score:
                g_best_score = p_best_scores[best_iter_idx]
                g_best_position = np.copy(p_best_positions[best_iter_idx])
            
            pct = int(((iteration + 1) / self.max_iter) * 100)
            progress_bar.progress(pct)
            status_text.text(f"🚀 High-Speed Swarm Running: Iteration {iteration+1}/{self.max_iter} | Best Cost: {g_best_score:.5f}")
            
        return g_best_position

# ==========================================
# USER INTERFACE
# ==========================================
st.title("🌦️ Weather Forecasting Prediction System")
st.markdown("### Framework: Optimized BPSO Feature Selection & BPNN Neural Network")
st.write("---")

# --- NEW: DYNAMIC CSV FILE UPLOADER SECTION ---
st.markdown("### 📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your weather forecasting metrics file (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV directly from memory
    df_raw = pd.read_csv(uploaded_file)
    st.success("📊 Dataset uploaded successfully!")
    
    # Show a brief structural preview of the uploaded data
    with st.expander("🔍 Preview Uploaded Data Structure"):
        st.dataframe(df_raw.head(5), use_container_width=True)

    # --- SIDEBAR CONFIGURATION CONTROLS ---
    st.sidebar.header("📊 Model Execution Configurations")

    all_columns = df_raw.columns.tolist()
    excluded = ['DATE', 'MONTH', 'date', 'month', 'Id', 'id']
    selectable_targets = [col for col in all_columns if col not in excluded]

    default_target = "BASEL_temp_mean" if "BASEL_temp_mean" in selectable_targets else selectable_targets[0]
    target_column = st.sidebar.selectbox("🎯 Select Target Column to Forecast", options=selectable_targets, index=selectable_targets.index(default_target) if default_target in selectable_targets else 0)

    st.sidebar.markdown("---")
    st.sidebar.header("⏳ Time-Series Settings")
    add_lags = st.sidebar.checkbox("Generate Historical Lag Features (t-1, t-2)", value=True)

    st.sidebar.header("⚙️ Optimized Swarm Sizes")
    swarm_size = st.sidebar.slider("Number of Swarm Particles", min_value=5, max_value=20, value=10, step=1)
    iterations = st.sidebar.slider("PSO Iteration Generations", min_value=5, max_value=25, value=10, step=1)

    st.sidebar.markdown("---")
    st.sidebar.header("🧠 Final BPNN Structure")
    hidden_layers = st.sidebar.text_input("Hidden Layer Topology", value="32, 16")

    try:
        layer_sizes = tuple(map(int, hidden_layers.split(',')))
    except:
        st.sidebar.error("Invalid syntax. Specify format as: 32, 16")
        st.stop()

    # --- PIPELINE RUNTIME EXECUTION ---
    if st.button("🚀 Execute Hybrid Pipeline Optimization"):
        st.write("### 🏃‍♂️ Pipeline Execution Progress")
        
        working_df = df_raw.copy()
        if add_lags:
            lag_dict = {}
            lag_dict[f"{target_column}_lag1"] = working_df[target_column].shift(1)
            lag_dict[f"{target_column}_lag2"] = working_df[target_column].shift(2)
            working_df = pd.concat([working_df, pd.DataFrame(lag_dict)], axis=1)
        
        # Clean up columns that cannot be cast to mathematical arrays
        cols_to_drop = [col for col in excluded if col in working_df.columns]
        working_df = working_df.drop(columns=cols_to_drop, errors='ignore')
        
        clean_df = working_df.dropna()
        features = [col for col in clean_df.columns if col != target_column]
        
        X_raw = clean_df[features].select_dtypes(include=[np.number]).values
        y_raw = clean_df[target_column].values
        
        # Guard clause for empty or corrupted feature matrices
        if X_raw.shape[1] == 0:
            st.error("The uploaded file does not contain valid numeric attributes to optimize.")
            st.stop()
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        pso_bar = st.progress(0)
        pso_text = st.empty()
        
        pso_engine = BPSOFeatureSelection(X_scaled, y_raw, num_particles=swarm_size, max_iter=iterations)
        optimal_mask = pso_engine.optimize(pso_bar, pso_text)
        selected_features = [features[i] for i, mask in enumerate(optimal_mask) if mask == 1]
        
        if len(selected_features) == 0:
            st.error("Optimization failed to capture a suitable feature subset.")
            st.stop()
            
        X_optimized = X_scaled[:, np.where(optimal_mask == 1)[0]]
        X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_raw, test_size=0.2, random_state=42)
        X_train_full, X_test_full, _, _ = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)
        
        st.success("🎉 Feature Selection Complete.")
        
        with st.spinner("Training Final Deep BPNN Forecasting Architecture..."):
            bpnn = MLPRegressor(hidden_layer_sizes=layer_sizes, activation='relu', solver='adam', learning_rate_init=0.01, max_iter=300, random_state=42, early_stopping=True)
            bpnn.fit(X_train, y_train)
            predictions = bpnn.predict(X_test)
            
        baseline_lr = SGDRegressor(max_iter=100, random_state=42)
        baseline_lr.fit(X_train_full, y_train)
        baseline_preds = baseline_lr.predict(X_test_full)
            
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        base_mae = mean_absolute_error(y_test, baseline_preds)
        base_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
        base_r2 = r2_score(y_test, baseline_preds)
        
        # --- RENDER ANALYTICS INTERFACE TABS ---
        st.write("### 📊 System Diagnostics & Visual Dashboards")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📉 Performance Metrics & Charts", 
            "🔍 Feature Space Reductions", 
            "🗺️ Pipeline Flowcharts", 
            "📋 Predictions Matrix"
        ])
        
        with tab1:
            st.markdown("#### Performance Comparison Matrix")
            metrics_comparison = pd.DataFrame({
                "Evaluation Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Coefficient of Determination"],
                "Proposed Hybrid System (PSO-BPNN)": [f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"],
                "Standard Baseline Tracker": [f"{base_mae:.4f}", f"{base_rmse:.4f}", f"{base_r2:.4f}"]
            })
            st.table(metrics_comparison)
            
            st.markdown("#### 📊 Comparative Error Bar Chart Analysis (Lower is Better)")
            chart_data = pd.DataFrame({
                "PSO-BPNN Model": [mae, rmse],
                "Baseline Model": [base_mae, base_rmse]
            }, index=["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"])
            
            st.bar_chart(chart_data, use_container_width=True)
            
        with tab2:
            st.markdown("#### Dimensionality Reduction Diagnostics")
            colA, colB = st.columns(2)
            colA.metric("Raw Input Features Space", len(features))
            colB.metric("Optimized Feature Set Used", len(selected_features))
            st.write(selected_features)
            
        with tab3:
            st.markdown("#### 🗺️ High-Resolution System Engineering Process Flow Diagrams")
            
            st.markdown("##### 1. End-to-End System Architecture Data Pipeline")
            st.graphviz_chart(generate_pipeline_flowchart(), use_container_width=True)
            
            st.write("---")
            
            st.markdown("##### 2. BPSO Optimization Evolutionary Control Flow")
            st.graphviz_chart(generate_algo_flowchart(selected_features), use_container_width=True)
            
        with tab4:
            st.markdown("#### Forecast Projections Review Matrix")
            comparison_df = pd.DataFrame({
                "Observed Target Values": y_test,
                "BPNN Forecast Projections": predictions,
                "Baseline Projections": baseline_preds,
                "Hybrid Error Residual Delta": y_test - predictions
            })
            st.dataframe(comparison_df.head(100), use_container_width=True)
    else:
        st.info("💡 Adjust your configurations in the sidebar drawer panel layout and hit execution run to view calculations, charts, and flowcharts.")
else:
    st.info("👋 Please upload a weather CSV file above to initialize the pipeline configuration controls and runtime engine.")