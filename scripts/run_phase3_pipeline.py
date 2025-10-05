"""
Phase 3 Pipeline: Model Development & Training
Train multiple models and track with MLflow
"""

import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.xgboost_model import XGBoostModel


def run_phase3_pipeline(data_path: str = "data/processed/feature_engineered_data.csv"):
    """Execute Phase 3: Model Training"""
    
    print("\n" + "="*70)
    print("  PHASE 3: MODEL DEVELOPMENT & TRAINING")
    print("="*70)
    
    # Load feature-engineered data
    print("\nLoading feature-engineered data...")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
    except FileNotFoundError:
        print(f"Error: Data not found at {data_path}")
        print("Please run Phase 2 first")
        return None
    
    # Setup MLflow
    mlflow_dir = "models/mlflow_runs"
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{os.path.abspath(mlflow_dir)}")
    mlflow.set_experiment("amazon_delivery_prediction")
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegressionModel(),
        'Random Forest': RandomForestModel(n_estimators=100, max_depth=10),
        'Gradient Boosting': GradientBoostingModel(n_estimators=100, learning_rate=0.1),
        'XGBoost': XGBoostModel(n_estimators=100, learning_rate=0.1)
    }
    
    results = []
    
    # Train each model
    for model_name, model in models.items():
        print("\n" + "="*70)
        print(f"Training: {model_name}")
        print("="*70)
        
        with mlflow.start_run(run_name=model_name):
            # Prepare data
            X_train, X_test, y_train, y_test = model.prepare_data(df)
            
            # Train
            model.train(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            
            # Cross-validate
            cv_results = model.cross_validate(X_train, y_train, cv=5)
            
            # Log to MLflow
            mlflow.log_params({
                'model_type': model_name,
                'n_features': len(model.feature_names)
            })
            
            mlflow.log_metrics({
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'cv_rmse': cv_results['cv_rmse_mean'],
                'cv_r2': cv_results['cv_r2_mean']
            })
            
            # Log model
            mlflow.sklearn.log_model(model.model, model_name.replace(' ', '_'))
            
            # Save model locally
            os.makedirs('models/trained_models', exist_ok=True)
            model_path = f"models/trained_models/{model_name.replace(' ', '_')}.pkl"
            model.save_model(model_path)
            
            # Feature importance
            importance = model.get_feature_importance()
            if importance is not None:
                print(f"\nTop 10 Important Features:")
                print(importance.head(10).to_string(index=False))
            
            # Store results
            results.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'CV_RMSE': cv_results['cv_rmse_mean'],
                'CV_R²': cv_results['cv_r2_mean']
            })
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    print(f"\n{results_df.to_string(index=False)}")
    
    # Save comparison
    results_df.to_csv('reports/model_comparison.csv', index=False)
    print(f"\nModel comparison saved to: reports/model_comparison.csv")
    
    # Best model
    best_model = results_df.iloc[0]['Model']
    best_rmse = results_df.iloc[0]['RMSE']
    best_r2 = results_df.iloc[0]['R²']
    
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_model}")
    print(f"  RMSE: {best_rmse:.2f}")
    print(f"  R²: {best_r2:.4f}")
    print(f"\nMLflow UI: mlflow ui --backend-store-uri {os.path.abspath(mlflow_dir)}")
    print("\nReady for Phase 4: Streamlit Application!")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 3: Model Training')
    parser.add_argument('--data-path', type=str,
                       default='data/processed/feature_engineered_data.csv',
                       help='Path to feature-engineered data')
    
    args = parser.parse_args()
    
    try:
        results = run_phase3_pipeline(args.data_path)
        
        if results is not None:
            print("\nPipeline executed successfully!")
            print("\nNext: python scripts/run_phase4_pipeline.py")
    
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)