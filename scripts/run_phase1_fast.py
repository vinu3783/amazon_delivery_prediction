"""Fast Phase 1 Pipeline - Data Cleaning Only"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner

def run_fast_pipeline(data_path: str = "data/raw/amazon_delivery.csv"):
    print("\n" + "="*70)
    print("  FAST PHASE 1: DATA CLEANING ONLY")
    print("="*70)
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    if df is None:
        print("Failed to load data")
        return None
    
    print(f"Loaded {len(df)} records")
    
    # Clean data
    print("\nCleaning data...")
    cleaner = DataCleaner(df)
    
    cleaner.strip_whitespace()
    cleaner.standardize_categories()
    cleaner.remove_duplicates()
    cleaner.handle_missing_values()
    cleaner.convert_datatypes()
    cleaner.handle_outliers(['Agent_Age', 'Agent_Rating', 'Delivery_Time'], method='iqr')
    
    cleaned_df = cleaner.get_cleaned_data()
    cleaner.print_cleaning_summary()
    
    # Save
    output_path = "data/processed/cleaned_data.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")
    print(f"Shape: {cleaned_df.shape}")
    print("\n🚀 Ready for Phase 2!")
    
    return cleaned_df

if __name__ == "__main__":
    run_fast_pipeline()