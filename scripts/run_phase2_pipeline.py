"""
Phase 2 Pipeline: Feature Engineering
Create and transform features for model training
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.feature_engineering import FeatureEngineer


def run_phase2_pipeline(data_path: str = "data/processed/cleaned_data.csv"):
    """Execute Phase 2: Feature Engineering"""
    
    print("\n" + "="*70)
    print("  PHASE 2: FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load cleaned data
    print("\nLoading cleaned data...")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
    except FileNotFoundError:
        print(f"Error: Cleaned data not found at {data_path}")
        print("Please run Phase 1 first: python scripts/run_phase1_pipeline.py")
        return None
    
    # Initialize Feature Engineer
    engineer = FeatureEngineer(df)
    
    # Step 1: Geospatial Features
    print("\n" + "="*70)
    print("Step 1: Geospatial Features")
    print("="*70)
    engineer.calculate_haversine_distance()
    engineer.create_distance_bins()
    
    # Step 2: Time Features
    print("\n" + "="*70)
    print("Step 2: Time-Based Features")
    print("="*70)
    engineer.extract_time_features()
    engineer.calculate_time_difference()
    
    # Step 3: Agent Features
    print("\n" + "="*70)
    print("Step 3: Agent Features")
    print("="*70)
    engineer.create_agent_features()
    
    # Step 4: Interaction Features
    print("\n" + "="*70)
    print("Step 4: Interaction Features")
    print("="*70)
    engineer.create_interaction_features()
    
    # After Step 5: Encode Categorical Features
# Add this:
    print("\n" + "="*70)
    print("Step 6: Handle Missing Values")
    print("="*70)
    engineer.handle_missing_values()

# Get engineered data
    engineered_df = engineer.get_engineered_data()
    
    # Print summary
    engineer.print_summary()
    
    # Save engineered data
    output_path = "data/processed/feature_engineered_data.csv"
    engineered_df.to_csv(output_path, index=False)
    print(f"\nFeature-engineered data saved to: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE!")
    print("="*70)
    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Final features: {len(engineered_df.columns)}")
    print(f"New features added: {len(engineered_df.columns) - len(df.columns)}")
    print("\nReady for Phase 3: Model Development!")
    
    return engineered_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 2: Feature Engineering')
    parser.add_argument('--data-path', type=str,
                       default='data/processed/cleaned_data.csv',
                       help='Path to cleaned data file')
    
    args = parser.parse_args()
    
    try:
        engineered_data = run_phase2_pipeline(args.data_path)
        
        if engineered_data is not None:
            print("\nPipeline executed successfully!")
            print("\nNext: python scripts/run_phase3_pipeline.py")
    
    except Exception as e:
        print(f"\nPipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)