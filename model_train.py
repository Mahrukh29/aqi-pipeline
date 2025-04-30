import pandas as pd
import numpy as np
import hopsworks
import joblib
import logging
import json
import time
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
HOPSWORKS_API_KEY_PATH = r"C:\Users\MAHRUKH BAIG\OneDrive\Desktop\AQI Prediction\secrets.json"
FEATURE_GROUP_NAME = "pm10_features"
FEATURE_NAMES_PATH = "feature_names.pkl"
MODEL_PATH_LGBM = "pm10_model_lgbm.pkl"
MODEL_PATH_XGB = "pm10_model_xgb.pkl"
MODEL_PATH_GB = "pm10_model_gb.pkl"
MODEL_PATH_RF = "pm10_model_rf.pkl"

def calculate_accuracy(y_true, y_pred, tolerance=10):
    within_tolerance = np.abs(y_true - y_pred) <= tolerance
    return np.mean(within_tolerance) * 100

def train_model():
    try:
        # Connect to Hopsworks
        logging.info("Connecting to Hopsworks...")
        with open(HOPSWORKS_API_KEY_PATH, 'r') as f:
            secrets = json.load(f)
        api_key = secrets["api_key"]
        project = hopsworks.login(api_key_value=api_key)
        feature_store = project.get_feature_store()

        # Check if the feature group exists
        logging.info(f"Checking if feature group '{FEATURE_GROUP_NAME}' exists...")
        feature_groups = feature_store.get_feature_groups(name=FEATURE_GROUP_NAME)
        if not feature_groups:
            logging.error(f"Feature group '{FEATURE_GROUP_NAME}' does not exist in Hopsworks.")
            raise ValueError(f"Feature group '{FEATURE_GROUP_NAME}' does not exist in Hopsworks. Please create it using create_features.py.")
        logging.info(f"Found feature group '{FEATURE_GROUP_NAME}' with versions: {[fg.version for fg in feature_groups]}")

        # Access the feature group
        logging.info(f"Accessing feature group '{FEATURE_GROUP_NAME}' (version 1)...")
        feature_group = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
        if not feature_group:
            logging.error(f"Feature group '{FEATURE_GROUP_NAME}' version 1 not found in Hopsworks.")
            raise ValueError(f"Feature group '{FEATURE_GROUP_NAME}' version 1 not found in Hopsworks.")

        # Validate the feature group
        logging.info("Validating feature group by reading a small sample...")
        sample_data = feature_group.select_all().show(1)
        if sample_data.empty:
            logging.error(f"Feature group '{FEATURE_GROUP_NAME}' is empty.")
            raise ValueError(f"Feature group '{FEATURE_GROUP_NAME}' is empty.")
        logging.info("Feature group validation successful.")

        # Set up or fetch the feature view
        feature_view_name = "pm10_prediction_view"
        desired_version = 1
        logging.info(f"Looking for feature view '{feature_view_name}'...")
        feature_view = None
        try:
            feature_views = feature_store.get_feature_views(name=feature_view_name)
            if feature_views:
                versions = [fv.version for fv in feature_views]
                latest_version = max(versions)
                logging.info(f"Found existing feature view '{feature_view_name}' with versions: {versions}. Attempting to use version {latest_version}...")
                feature_view = feature_store.get_feature_view(name=feature_view_name, version=latest_version)
                sample_data = feature_view.get_batch_data(start_time=0, end_time=1)
                logging.info(f"Feature view version {latest_version} is valid.")
                desired_version = latest_version
            else:
                logging.warning(f"Could not retrieve feature views. Proceeding to create a new one...")
        except Exception as e:
            logging.warning(f"Feature view version {latest_version} is invalid or not found: {str(e)}. Proceeding to create a new one...")

        if not feature_view:
            logging.info(f"Creating a new feature view '{feature_view_name}' with version {desired_version}...")
            feature_view = feature_store.create_feature_view(
                name=feature_view_name,
                version=desired_version,
                query=feature_group.select_all(),
                description="Feature view for PM10 prediction"
            )
            logging.info(f"Feature view '{feature_view_name}' (version {desired_version}) created successfully.")
        time.sleep(10)

        # Load training data
        training_datasets = feature_view.get_training_datasets()
        if not training_datasets:
            logging.info("No training dataset found. Creating a new one...")
            feature_view.create_training_data(
                description="Training dataset for PM10 prediction",
                data_format="parquet",
                write_options={"wait_for_job": True}
            )
            training_dataset_version = 1
        else:
            training_dataset_version = max([td.version for td in training_datasets])
            logging.info(f"Using existing training dataset version: {training_dataset_version}")

        train_data = feature_view.get_training_data(training_dataset_version=training_dataset_version)[0]
        if train_data.empty:
            logging.error("Training data is empty. Run create_features.py to populate the feature group.")
            raise ValueError("Training data is empty.")

        # Clean up the data
        duplicates = train_data.duplicated(subset=['datetime', 'record_id']).sum()
        logging.info(f"Found {duplicates} duplicate rows based on datetime and record_id.")
        if duplicates > 0:
            train_data = train_data.drop_duplicates(subset=['datetime', 'record_id'], keep='last')
            logging.info(f"Removed duplicates. New row count: {len(train_data)}")

        pm10_upper_limit = 400
        outliers = train_data[train_data['pm10'] > pm10_upper_limit]
        logging.info(f"Found {len(outliers)} rows with PM10 > {pm10_upper_limit} µg/m³.")
        train_data = train_data[train_data['pm10'] <= pm10_upper_limit]
        logging.info(f"After removing outliers, row count: {len(train_data)}")

        logging.info(f"Loaded {len(train_data)} rows of training data.")
        logging.info(f"PM10 stats:\n{train_data['pm10'].describe()}")

        # Feature Engineering
        train_data['pm25_no2'] = train_data['pm25'] * train_data['no2']
        train_data['temp_wind'] = train_data['temperature'] * train_data['wind_speed']
        logging.info("Added interaction terms: pm25_no2, temp_wind")

        train_data = train_data.sort_values('datetime')
        train_data['lag_2_pm25'] = train_data['pm25'].shift(2)
        train_data = train_data.dropna()
        logging.info("Added lagged features: lag_2_pm25")
        logging.info(f"Row count after adding lagged features: {len(train_data)}")

        train_data['pm25_o3'] = train_data['pm25'] * train_data['o3']
        logging.info("Added interaction term: pm25_o3")

        # Prepare features and target
        columns_to_drop = [
            'pm10', 'datetime', 'record_id', 'lag_1_pm10', 'rolling_pm10_mean',
            'rolling_pm10_std', 'pm10_change', 'pm25_pm10_ratio', 'hour',
            'precipitation', 'hour_sin', 'hour_cos', 'lag_2_pm25', 'day_of_week', 'heat_index'  # Dropped low-importance features
        ]
        features = train_data.drop(columns=columns_to_drop)
        target = train_data['pm10']

        # Add feature selection based on correlation
        correlation_matrix = features.corr()
        high_corr_pairs = [(i, j) for i in correlation_matrix.columns for j in correlation_matrix.columns
                           if i < j and abs(correlation_matrix.loc[i, j]) > 0.9]
        for feat1, feat2 in high_corr_pairs:
            logging.info(f"Dropping {feat2} due to high correlation with {feat1} (|corr| > 0.9)")
            if feat2 in features.columns:
                features = features.drop(columns=[feat2])

        features = features.fillna(features.median())
        target = target.fillna(target.median())

        feature_names = features.columns.tolist()
        joblib.dump(feature_names, FEATURE_NAMES_PATH)
        logging.info(f"Features used for training: {feature_names}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train and evaluate models
        model_metrics = {}

        # LightGBM with updated parameters
        logging.info("Training LightGBM model with hyperparameter tuning and early stopping...")
        lgbm_base = lgb.LGBMRegressor(random_state=42, force_col_wise=True)
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [5, 8],
            'min_child_samples': [1],
            'min_child_weight': [0.0001],
            'reg_alpha': [0.0, 0.1],
            'reg_lambda': [0.0, 0.1],
            'num_leaves': [30, 60],
            'min_split_gain': [0.0]
        }
        grid_search = GridSearchCV(lgbm_base, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2',
                        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        lgbm_model = grid_search.best_estimator_
        logging.info(f"Best LightGBM parameters: {grid_search.best_params_}")

        lgbm_predictions = lgbm_model.predict(X_test)
        lgbm_r2 = r2_score(y_test, lgbm_predictions)
        lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
        lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
        lgbm_accuracy = calculate_accuracy(y_test, lgbm_predictions)
        model_metrics['LightGBM'] = {'r2': lgbm_r2, 'mse': lgbm_mse, 'mae': lgbm_mae, 'accuracy': lgbm_accuracy}
        logging.info(f"LightGBM Results - R²: {lgbm_r2:.4f}, Accuracy (±10 µg/m³): {lgbm_accuracy:.2f}%")

        lgbm_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': lgbm_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        logging.info(f"LightGBM Feature Importance:\n{lgbm_importance}")

        # XGBoost
        logging.info("Training XGBoost model with hyperparameter tuning...")
        xgb_base = XGBRegressor(random_state=42)
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        grid_search_xgb = GridSearchCV(xgb_base, param_grid_xgb, cv=3, scoring='r2', n_jobs=-1)
        grid_search_xgb.fit(X_train, y_train)
        xgb_model = grid_search_xgb.best_estimator_
        logging.info(f"Best XGBoost parameters: {grid_search_xgb.best_params_}")

        xgb_predictions = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_predictions)
        xgb_mse = mean_squared_error(y_test, xgb_predictions)
        xgb_mae = mean_absolute_error(y_test, xgb_predictions)
        xgb_accuracy = calculate_accuracy(y_test, xgb_predictions)
        model_metrics['XGBoost'] = {'r2': xgb_r2, 'mse': xgb_mse, 'mae': xgb_mae, 'accuracy': xgb_accuracy}
        logging.info(f"XGBoost Results - R²: {xgb_r2:.4f}, Accuracy (±10 µg/m³): {xgb_accuracy:.2f}%")

        xgb_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        logging.info(f"XGBoost Feature Importance:\n{xgb_importance}")

        # Gradient Boosting
        logging.info("Training Gradient Boosting model with hyperparameter tuning...")
        gb_base = GradientBoostingRegressor(random_state=42)
        param_grid_gb = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        grid_search_gb = GridSearchCV(gb_base, param_grid_gb, cv=3, scoring='r2', n_jobs=-1)
        grid_search_gb.fit(X_train, y_train)
        gb_model = grid_search_gb.best_estimator_
        logging.info(f"Best Gradient Boosting parameters: {grid_search_gb.best_params_}")

        gb_predictions = gb_model.predict(X_test)
        gb_r2 = r2_score(y_test, gb_predictions)
        gb_mse = mean_squared_error(y_test, gb_predictions)
        gb_mae = mean_absolute_error(y_test, gb_predictions)
        gb_accuracy = calculate_accuracy(y_test, gb_predictions)
        model_metrics['GradientBoosting'] = {'r2': gb_r2, 'mse': gb_mse, 'mae': gb_mae, 'accuracy': gb_accuracy}
        logging.info(f"Gradient Boosting Results - R²: {gb_r2:.4f}, Accuracy (±10 µg/m³): {gb_accuracy:.2f}%")

        gb_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': gb_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        logging.info(f"Gradient Boosting Feature Importance:\n{gb_importance}")

        # Random Forest
        logging.info("Training Random Forest model with hyperparameter tuning...")
        rf_base = RandomForestRegressor(random_state=42)
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        grid_search_rf = GridSearchCV(rf_base, param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        rf_model = grid_search_rf.best_estimator_
        logging.info(f"Best Random Forest parameters: {grid_search_rf.best_params_}")

        rf_predictions = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_predictions)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_mae = mean_absolute_error(y_test, rf_predictions)
        rf_accuracy = calculate_accuracy(y_test, rf_predictions)
        model_metrics['RandomForest'] = {'r2': rf_r2, 'mse': rf_mse, 'mae': rf_mae, 'accuracy': rf_accuracy}
        logging.info(f"Random Forest Results - R²: {rf_r2:.4f}, Accuracy (±10 µg/m³): {rf_accuracy:.2f}%")

        rf_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        logging.info(f"Random Forest Feature Importance:\n{rf_importance}")

        # Ensemble with adjusted weights
        logging.info("Creating ensemble model with weighted predictions...")
        weights = {'LightGBM': 0.5, 'XGBoost': 0.3, 'GradientBoosting': 0.1, 'RandomForest': 0.1}  # Adjusted weights
        ensemble_predictions = (
            weights['LightGBM'] * lgbm_predictions +
            weights['XGBoost'] * xgb_predictions +
            weights['GradientBoosting'] * gb_predictions +
            weights['RandomForest'] * rf_predictions
        )
        ensemble_r2 = r2_score(y_test, ensemble_predictions)
        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
        ensemble_accuracy = calculate_accuracy(y_test, ensemble_predictions)
        model_metrics['Ensemble'] = {'r2': ensemble_r2, 'mse': ensemble_mse, 'mae': ensemble_mae, 'accuracy': ensemble_accuracy}
        logging.info(f"Ensemble Results - R²: {ensemble_r2:.4f}, Accuracy (±10 µg/m³): {ensemble_accuracy:.2f}%")

        # Cross-validation for the ensemble
        logging.info("Performing 5-fold cross-validation for the ensemble model...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ensemble_cv_r2_scores = []
        ensemble_cv_accuracy_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            X_fold_train, X_fold_val = features.iloc[train_idx], features.iloc[val_idx]
            y_fold_train, y_fold_val = target.iloc[train_idx], target.iloc[val_idx]

            # Train LightGBM on the fold
            lgbm_model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], 
                          eval_metric='l2', callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
            lgbm_fold_pred = lgbm_model.predict(X_fold_val)

            # Train XGBoost on the fold
            xgb_model.fit(X_fold_train, y_fold_train)
            xgb_fold_pred = xgb_model.predict(X_fold_val)

            # Train Gradient Boosting on the fold
            gb_model.fit(X_fold_train, y_fold_train)
            gb_fold_pred = gb_model.predict(X_fold_val)

            # Train Random Forest on the fold
            rf_model.fit(X_fold_train, y_fold_train)
            rf_fold_pred = rf_model.predict(X_fold_val)

            # Ensemble predictions for the fold
            ensemble_fold_pred = (
                weights['LightGBM'] * lgbm_fold_pred +
                weights['XGBoost'] * xgb_fold_pred +
                weights['GradientBoosting'] * gb_fold_pred +
                weights['RandomForest'] * rf_fold_pred
            )

            # Compute metrics for the fold
            fold_r2 = r2_score(y_fold_val, ensemble_fold_pred)
            fold_accuracy = calculate_accuracy(y_fold_val, ensemble_fold_pred)
            ensemble_cv_r2_scores.append(fold_r2)
            ensemble_cv_accuracy_scores.append(fold_accuracy)

        logging.info(f"Ensemble 5-fold CV R² scores: {ensemble_cv_r2_scores}")
        logging.info(f"Ensemble 5-fold CV R² mean: {np.mean(ensemble_cv_r2_scores):.4f}, std: {np.std(ensemble_cv_r2_scores):.4f}")
        logging.info(f"Ensemble 5-fold CV Accuracy scores: {ensemble_cv_accuracy_scores}")
        logging.info(f"Ensemble 5-fold CV Accuracy mean: {np.mean(ensemble_cv_accuracy_scores):.2f}%, std: {np.std(ensemble_cv_accuracy_scores):.2f}")

        # Log model performance summary
        logging.info("\n=== Model Performance Summary ===")
        summary_df = pd.DataFrame.from_dict(model_metrics, orient='index')[['r2', 'accuracy']]
        summary_df.columns = ['R²', 'Accuracy (±10 µg/m³)']
        summary_df['R²'] = summary_df['R²'].round(4)
        summary_df['Accuracy (±10 µg/m³)'] = summary_df['Accuracy (±10 µg/m³)'].round(2)
        logging.info(f"\n{summary_df.to_string()}")

        # Save the best model
        best_model_name = max(model_metrics, key=lambda x: model_metrics[x]['r2'])
        best_metrics = model_metrics[best_model_name]

        if best_model_name == 'LightGBM':
            best_model = lgbm_model
            best_model_path = MODEL_PATH_LGBM
        elif best_model_name == 'XGBoost':
            best_model = xgb_model
            best_model_path = MODEL_PATH_XGB
        elif best_model_name == 'GradientBoosting':
            best_model = gb_model
            best_model_path = MODEL_PATH_GB
        elif best_model_name == 'RandomForest':
            best_model = rf_model
            best_model_path = MODEL_PATH_RF
        elif best_model_name == 'Ensemble':
            logging.info("Ensemble model selected as best. Saving individual models.")
            best_model = None
            best_model_path = MODEL_PATH_LGBM

        logging.info(f"Best model: {best_model_name} with R²: {best_metrics['r2']:.4f}, Accuracy: {best_metrics['accuracy']:.2f}%")

        if best_model:
            model_registry = project.get_model_registry()
            joblib.dump(best_model, best_model_path)
            model = model_registry.python.create_model(
                name="pm10_model",
                metrics={
                    "test_r2": best_metrics['r2'],
                    "test_mse": best_metrics['mse'],
                    "test_mae": best_metrics['mae'],
                    "test_accuracy": best_metrics['accuracy']
                },
                description=f"Best model ({best_model_name}) for PM10 prediction"
            )
            model.save(best_model_path)
            logging.info("Best model saved to Hopsworks Model Registry.")

        # Save all individual models locally
        joblib.dump(lgbm_model, MODEL_PATH_LGBM)
        joblib.dump(xgb_model, MODEL_PATH_XGB)
        joblib.dump(gb_model, MODEL_PATH_GB)
        joblib.dump(rf_model, MODEL_PATH_RF)
        logging.info("All individual models saved locally.")

        hopsworks.logout()
        logging.info("Hopsworks connection closed.")

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        hopsworks.logout()
        raise

if __name__ == "__main__":
    logging.info("Starting the PM10 prediction model training pipeline...")
    train_model()
    logging.info("Training pipeline completed successfully!")