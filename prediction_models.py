import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np

file_path = "train.txt"
unit_col = "unit_number"
engine_col = "time_in_cycles"

def read_file(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""

    df = pd.read_csv(file_path, delimiter=r'\s+', engine="python", header=None)

    # add column names - works for the original NASA C-MAPSS dataset format

    cols = [unit_col, engine_col, "operational_setting_1", "operational_setting_2", "operational_setting_3"]

    for i in range(1, len(df.columns)-4):
        cols.append("sensor_measurement_"+ str(i))
    df.columns=cols

    # add failing cycle and remaining useful life (RUL) columns

    df["fail_cycle"] = df.groupby(unit_col)[engine_col].transform("max")
    df["RUL"] = df["fail_cycle"] - df[engine_col]

    return df


def create_test_train_datasets(df, test_size=0.3, random_state=42):
    # only split by engine
    engines = df[unit_col].unique()

    train_engines, valid_engines = train_test_split(engines, test_size=test_size, random_state=random_state)

    train_df = df[df[unit_col].isin(train_engines)].copy()
    valid_df = df[df[unit_col].isin(valid_engines)].copy()

    # check there's no engine overlap between train and test

    leakage = len(set(train_engines) & set(valid_engines))
    print(f"overlap in train and test engines: {leakage}")

    # select features, exclude RUL - keep original column order
    feature_cols = [c for c in df.columns if c not in ["fail_cycle", "RUL"]]

    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["RUL"]

    # add lagged features
    X_train, X_valid = add_lagged_features(X_train, X_valid, lag_steps=10)

    # add rolling statistics
    X_train, X_valid = add_rolling_statistics(X_train, X_valid)

    # drop rows with NaN values created by lag features
    X_train.dropna(inplace=True)
    X_valid.dropna(inplace=True)

    # Align y with X after dropping NaN rows
    y_train = y_train.loc[X_train.index]
    y_valid = y_valid.loc[X_valid.index]

    # Remove unit_number before scaling (it's used for groupby but shouldn't be a feature)
    X_train = X_train.drop(columns=[unit_col])
    X_valid = X_valid.drop(columns=[unit_col])

    # Get updated feature columns after adding lag and rolling features
    feature_cols = list(X_train.columns)
    
    return X_train, y_train, X_valid, y_valid, feature_cols

def add_lagged_features(train_df, valid_df, lag_steps):
    #. jatka tätä ja rolling statistics lisäksi
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor')]
    lags = [i for i in range(1, lag_steps+1)]
    train_lag_cols = {}
    valid_lag_cols = {}
    for lag in lags:
        for col in sensor_cols:
            col_name = f'{col}_lag_{lag}'
            train_lag_cols[col_name] = train_df.groupby('unit_number')[col].shift(lag)
            valid_lag_cols[col_name] = valid_df.groupby('unit_number')[col].shift(lag)
    # Concat all new columns at once
    train_df = pd.concat([train_df, pd.DataFrame(train_lag_cols)], axis=1)
    valid_df = pd.concat([valid_df, pd.DataFrame(valid_lag_cols)], axis=1)
    return train_df, valid_df

def add_rolling_statistics(train_df, valid_df):
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor') and '_lag_' not in col and '_roll_' not in col]
    # Rolling statistics - optimized to avoid fragmentation
    window = 10
    stats = ['mean', 'std', 'min', 'max']

    train_roll_cols = {}
    valid_roll_cols = {}

    for stat in stats:
        for col in sensor_cols:
            col_name = f'{col}_roll_{stat}_{window}'
            train_roll_cols[col_name] = (
                train_df.groupby('unit_number')[col]
                .rolling(window=window, min_periods=1)
                .agg(stat)
                .reset_index(level=0, drop=True)
            )
            valid_roll_cols[col_name] = (
                valid_df.groupby('unit_number')[col]
                .rolling(window=window, min_periods=1)
                .agg(stat)
                .reset_index(level=0, drop=True)
            )
    
    # Concat all new columns at once (avoids fragmentation)
    train_df = pd.concat([train_df, pd.DataFrame(train_roll_cols)], axis=1)
    valid_df = pd.concat([valid_df, pd.DataFrame(valid_roll_cols)], axis=1)
    
    return train_df, valid_df

def scaling(X_train, X_valid, feature_cols):
    scaler = StandardScaler()

    # Keep feature names for later use
    feature_names = list(feature_cols)

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Convert back to DataFrame to preserve column names AND index
    X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_valid = pd.DataFrame(X_valid_scaled, columns=feature_names, index=X_valid.index)
    
    return X_train, X_valid


def run_xgboost_model(X_train, y_train):
    model_xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.6,
        gamma=0.1, # if lagged features
        reg_alpha=1.0, # if lagged features
        reg_lambda=2.0, # if lagged features
        objective="reg:squarederror",
        #n_jobs=-1, #run on all cores, speeds up
        random_state=42,
    )

    model_xgb.fit(X_train, y_train)
    return model_xgb

def print_statistics(model, X_train, y_train, X_valid, y_valid):
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error

    xgb.plot_importance(model, max_num_features=20)
    plt.show()
    # naive mean-predictor baseline
    mean_baseline = y_train.mean()
    baseline_pred_train = np.full_like(y_train, mean_baseline)
    baseline_pred_valid = np.full_like(y_valid, mean_baseline)

    # compare values to baseline
    baseline_train_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred_train))
    baseline_test_rmse = np.sqrt(mean_squared_error(y_valid, baseline_pred_valid))

    print(f"Baseline train RMSE: {baseline_train_rmse:.2f}")
    print(f"Baseline test RMSE: {baseline_test_rmse:.2f}")

    # validate on internal test set
    y_pred_valid = model.predict(X_valid)
    rmse_test = root_mean_squared_error(y_valid, y_pred_valid)
    mae_test  = mean_absolute_error(y_valid, y_pred_valid)
    r2_test = r2_score(y_valid, y_pred_valid)
    print(f"\nInternal test RMSE: {rmse_test:.2f}")
    print(f"Internal test MAE: {mae_test:.2f}")
    print(f"Internal test R2: {r2_test:.2f}")

    # validate on internal train set
    y_pred_tr = model.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_pred_tr)
    mae_train  = mean_absolute_error(y_train, y_pred_tr)
    r2_train = r2_score(y_train, y_pred_tr)
    print(f"\nInternal train RMSE: {rmse_train:.2f}")
    print(f"Internal train MAE: {mae_train:.2f}")
    print(f"Internal train R2: {r2_train:.2f}")

def main():
    # read data
    df = read_file(file_path)

    # create train and test datasets
    X_train, y_train, X_valid, y_valid, features = create_test_train_datasets(df)

    # scale data
    X_train, X_valid = scaling(X_train, X_valid, features)

    # train model
    model = run_xgboost_model(X_train, y_train)

    # print statistics
    print_statistics(model, X_train, y_train, X_valid, y_valid)

if __name__ == "__main__":
    main()