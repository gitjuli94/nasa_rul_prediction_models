import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error

from prediction_models import (
    read_file, 
    scaling, 
    create_test_train_datasets, 
    run_xgboost_model,
    file_path
)

st.set_page_config(page_title="NASA RUL Prediction", layout="wide")

st.title("🚀 NASA Engine RUL Prediction")
st.markdown("Predicting Remaining Useful Life (RUL) using XGBoost")

# Sidebar
st.sidebar.header("Model Configuration")
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3, 0.05)

if st.sidebar.button("🔄 Run Model", type="primary"):
    with st.spinner("Loading and preprocessing data..."):
        # Read and scale data
        df = read_file(file_path)
        
        # Display data info
        st.subheader("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Features", f"{len(df.columns)}")
        col3.metric("Unique Engines", f"{df['unit_number'].nunique()}")
        
        st.dataframe(df.head(10), use_container_width=True)
    
    with st.spinner("Creating train/test split and scaling..."):
        X_train, y_train, X_valid, y_valid, features = create_test_train_datasets(df, test_size=test_size)
        
        # Store unscaled data for display before scaling
        X_train_unscaled = X_train.copy()
        X_valid_unscaled = X_valid.copy()
        
        X_train, X_valid = scaling(X_train, X_valid, features)
        
        st.subheader("📂 Train/Test Split")
        col1, col2 = st.columns(2)
        col1.metric("Training Samples", f"{len(X_train):,}")
        col2.metric("Validation Samples", f"{len(X_valid):,}")
        
        # Display engineered features DataFrame
        st.subheader("🔧 Engineered Features (Lag + Rolling Statistics)")
        
        # Show feature counts
        lag_cols = [c for c in X_train_unscaled.columns if '_lag_' in c]
        roll_cols = [c for c in X_train_unscaled.columns if '_roll_' in c]
        original_cols = [c for c in X_train_unscaled.columns if '_lag_' not in c and '_roll_' not in c]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Features", f"{len(X_train_unscaled.columns)}")
        col2.metric("Original Features", f"{len(original_cols)}")
        col3.metric("Lag Features", f"{len(lag_cols)}")
        col4.metric("Rolling Features", f"{len(roll_cols)}")
        
        # Tabs for different feature groups
        tab1, tab2, tab3, tab4 = st.tabs(["📋 All Features", "📊 Original", "⏪ Lag Features", "📈 Rolling Stats"])
        
        with tab1:
            st.dataframe(X_train_unscaled.head(10), use_container_width=True)
        
        with tab2:
            st.dataframe(X_train_unscaled[original_cols].head(10), use_container_width=True)
        
        with tab3:
            if lag_cols:
                st.dataframe(X_train_unscaled[lag_cols].head(10), use_container_width=True)
            else:
                st.info("No lag features found")
        
        with tab4:
            if roll_cols:
                st.dataframe(X_train_unscaled[roll_cols].head(10), use_container_width=True)
            else:
                st.info("No rolling statistics features found")
    
    with st.spinner("Training XGBoost model..."):
        model = run_xgboost_model(X_train, y_train)
        st.success("✅ Model trained successfully!")
    
    # Calculate predictions and metrics
    st.subheader("📈 Model Performance")
    
    # Baseline metrics
    mean_baseline = y_train.mean()
    baseline_pred_train = np.full_like(y_train, mean_baseline)
    baseline_pred_valid = np.full_like(y_valid, mean_baseline)
    baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
    baseline_test_rmse = np.sqrt(mean_squared_error(y_valid, baseline_pred_valid))
    
    # Model predictions
    y_pred_train = model.predict(X_train)
    y_pred_valid = model.predict(X_valid)
    
    # Training metrics
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Validation metrics
    rmse_valid = root_mean_squared_error(y_valid, y_pred_valid)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    r2_valid = r2_score(y_valid, y_pred_valid)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Baseline (Mean Predictor)**")
        st.metric("Train RMSE", f"{baseline_train_rmse:.2f}")
        st.metric("Test RMSE", f"{baseline_test_rmse:.2f}")
    
    with col2:
        st.markdown("**Training Set**")
        st.metric("RMSE", f"{rmse_train:.2f}", delta=f"{baseline_train_rmse - rmse_train:.2f} vs baseline")
        st.metric("MAE", f"{mae_train:.2f}")
        st.metric("R²", f"{r2_train:.3f}")
    
    with col3:
        st.markdown("**Validation Set**")
        st.metric("RMSE", f"{rmse_valid:.2f}", delta=f"{baseline_test_rmse - rmse_valid:.2f} vs baseline")
        st.metric("MAE", f"{mae_valid:.2f}")
        st.metric("R²", f"{r2_valid:.3f}")
    
    # Feature importance plot
    st.subheader("🎯 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=20, ax=ax)
    ax.set_title('Top 20 Feature Importances')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Actual vs Predicted plot
    st.subheader("🔍 Actual vs Predicted RUL")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set
    axes[0].scatter(y_train, y_pred_train, alpha=0.3, s=10)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual RUL')
    axes[0].set_ylabel('Predicted RUL')
    axes[0].set_title('Training Set')
    
    # Validation set
    axes[1].scatter(y_valid, y_pred_valid, alpha=0.3, s=10)
    axes[1].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual RUL')
    axes[1].set_ylabel('Predicted RUL')
    axes[1].set_title('Validation Set')
    
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("👈 Click 'Run Model' in the sidebar to start the prediction pipeline")
