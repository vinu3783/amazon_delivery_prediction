"""
Phase 1 Pipeline: Data Analysis & Cleaning
Complete pipeline for data loading, cleaning, and exploratory analysis
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path (fix for import issues)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.visualization.exploratory_analysis import ExploratoryAnalysis
from src.visualization.outlier_analysis import OutlierAnalyzer


def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'reports/eda_plots',
        'reports/outlier_analysis',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def run_phase1_pipeline(data_path: str = "data/raw/amazon_delivery.csv"):
    """
    Execute Phase 1: Data Analysis & Cleaning Pipeline
    
    Args:
        data_path (str): Path to raw data file
    """
    print("\n" + "="*70)
    print("  PHASE 1: DATA ANALYSIS & CLEANING PIPELINE")
    print("="*70)
    
    # Step 1: Create directories
    print("\n📁 Step 1: Setting up directories...")
    create_directories()
    
    # Step 2: Load data
    print("\n📊 Step 2: Loading data...")
    print("-"*70)
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    if df is None:
        print("❌ Failed to load data. Exiting pipeline.")
        return None
    
    # Display data summary
    loader.display_summary()
    loader.check_duplicates()
    
    # Check missing values
    missing_summary = loader.get_missing_value_summary()
    if not missing_summary.empty:
        print("\n⚠️  Missing Values Detected:")
        print(missing_summary.to_string(index=False))
    
    # Step 3: Data Cleaning
    print("\n" + "="*70)
    print("🧹 Step 3: Data Cleaning")
    print("="*70)
    
    cleaner = DataCleaner(df)
    
    # Perform cleaning operations
    cleaner.strip_whitespace()
    cleaner.standardize_categories()
    cleaner.remove_duplicates()
    cleaner.handle_missing_values(strategy={'numerical': 'median', 'categorical': 'mode'})
    cleaner.convert_datatypes()
    
    # Handle outliers with IQR method
    outlier_columns = ['Agent_Age', 'Agent_Rating', 'Delivery_Time']
    cleaner.handle_outliers(outlier_columns, method='iqr', threshold=1.5)
    
    # Validate data
    cleaner.validate_data_ranges()
    
    # Get cleaned data
    cleaned_df = cleaner.get_cleaned_data()
    
    # Print cleaning summary
    cleaner.print_cleaning_summary()
    
    # Save cleaned data
    cleaned_data_path = "data/processed/cleaned_data.csv"
    cleaned_df.to_csv(cleaned_data_path, index=False)
    print(f"\n💾 Cleaned data saved to: {cleaned_data_path}")
    
    # Step 4: Exploratory Data Analysis
    print("\n" + "="*70)
    print("📊 Step 4: Exploratory Data Analysis")
    print("="*70)
    
    eda = ExploratoryAnalysis(cleaned_df)
    
    # Run EDA analyses
    eda_output_dir = "reports/eda_plots"
    
    print("\n📈 Running univariate analysis...")
    eda.univariate_analysis_numerical(save_path=eda_output_dir)
    eda.univariate_analysis_categorical(save_path=eda_output_dir)
    
    print("\n📉 Running target variable analysis...")
    eda.target_distribution_analysis(target='Delivery_Time', save_path=eda_output_dir)
    
    print("\n📊 Running bivariate analysis...")
    eda.bivariate_analysis(target='Delivery_Time', save_path=eda_output_dir)
    
    print("\n🔗 Running correlation analysis...")
    eda.correlation_analysis(save_path=eda_output_dir)
    
    print("\n⏰ Running time-based analysis...")
    eda.time_based_analysis(save_path=eda_output_dir)
    
    print("\n💡 Generating insights...")
    insights = eda.generate_insights()
    
    # Step 5: Outlier Analysis
    print("\n" + "="*70)
    print("🔍 Step 5: Advanced Outlier Analysis")
    print("="*70)
    
    outlier_analyzer = OutlierAnalyzer(cleaned_df)
    
    # Analyze outliers using multiple methods
    outlier_results = outlier_analyzer.analyze_all_outliers(methods=['iqr', 'zscore'])
    
    # Visualize outliers
    outlier_output_dir = "reports/outlier_analysis"
    print("\n📊 Generating outlier visualizations...")
    
    for col in ['Delivery_Time', 'Agent_Age', 'Agent_Rating']:
        outlier_analyzer.visualize_outliers(col, method='iqr', save_path=outlier_output_dir)
        outlier_analyzer.compare_outlier_methods(col, save_path=outlier_output_dir)
    
    # Get outlier summary
    outlier_summary = outlier_analyzer.get_outlier_summary()
    print("\n📋 Outlier Analysis Summary:")
    print(outlier_summary.to_string(index=False))
    
    # Step 6: Generate Final Report
    print("\n" + "="*70)
    print("📝 Step 6: Generating Final Report")
    print("="*70)
    
    report = generate_phase1_report(df, cleaned_df, insights, outlier_summary)
    
    report_path = "reports/phase1_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Phase 1 report saved to: {report_path}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print(f"  • Original records: {len(df)}")
    print(f"  • Cleaned records: {len(cleaned_df)}")
    print(f"  • Records removed: {len(df) - len(cleaned_df)}")
    print(f"  • Missing values handled: ✓")
    print(f"  • Outliers analyzed: ✓")
    print(f"  • EDA visualizations: ✓")
    print(f"\n📁 Output locations:")
    print(f"  • Cleaned data: {cleaned_data_path}")
    print(f"  • EDA plots: {eda_output_dir}/")
    print(f"  • Outlier analysis: {outlier_output_dir}/")
    print(f"  • Final report: {report_path}")
    print("\n🚀 Ready for Phase 2: Feature Engineering!")
    
    return cleaned_df


def generate_phase1_report(original_df, cleaned_df, insights, outlier_summary):
    """Generate comprehensive Phase 1 report"""
    
    report = f"""
{'='*80}
PHASE 1: DATA ANALYSIS & CLEANING REPORT
Amazon Delivery Time Prediction Project
{'='*80}

1. DATA OVERVIEW
{'-'*80}
Original Dataset:
  - Records: {len(original_df)}
  - Features: {len(original_df.columns)}
  - Size: {original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Cleaned Dataset:
  - Records: {len(cleaned_df)}
  - Features: {len(cleaned_df.columns)}
  - Size: {cleaned_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
  - Records removed: {len(original_df) - len(cleaned_df)}

2. DATA QUALITY ASSESSMENT
{'-'*80}
Missing Values (Original):
{original_df.isnull().sum().to_string()}

Missing Values (After Cleaning):
{cleaned_df.isnull().sum().to_string()}

Duplicate Rows:
  - Original: {original_df.duplicated().sum()}
  - After cleaning: {cleaned_df.duplicated().sum()}

3. FEATURE SUMMARY
{'-'*80}
Numerical Features:
{cleaned_df.select_dtypes(include=['number']).describe().to_string()}

Categorical Features:
"""
    
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        report += f"\n{col}:\n"
        report += f"  Unique values: {cleaned_df[col].nunique()}\n"
        value_counts = cleaned_df[col].value_counts().head(5)
        report += f"{value_counts.to_string()}\n"
    
    report += f"""

4. KEY INSIGHTS
{'-'*80}
"""
    
    if 'delivery_time' in insights:
        report += f"""
Delivery Time Analysis:
  - Average: {insights['delivery_time']['average']:.2f} minutes
  - Median: {insights['delivery_time']['median']:.2f} minutes
  - Common range: {insights['delivery_time']['most_common_range']}
"""
    
    if 'weather_impact' in insights:
        report += "\nWeather Impact on Delivery Time:\n"
        for weather, data in insights['weather_impact']['mean'].items():
            count = insights['weather_impact']['count'].get(weather, 0)
            report += f"  - {weather}: {data:.2f} min (n={count})\n"
    
    if 'traffic_impact' in insights:
        report += "\nTraffic Impact on Delivery Time:\n"
        for traffic, data in insights['traffic_impact']['mean'].items():
            count = insights['traffic_impact']['count'].get(traffic, 0)
            report += f"  - {traffic}: {data:.2f} min (n={count})\n"
    
    if 'vehicle_efficiency' in insights:
        report += "\nVehicle Efficiency:\n"
        for vehicle, data in insights['vehicle_efficiency']['mean'].items():
            count = insights['vehicle_efficiency']['count'].get(vehicle, 0)
            report += f"  - {vehicle}: {data:.2f} min (n={count})\n"
    
    report += f"""

5. OUTLIER ANALYSIS
{'-'*80}
{outlier_summary.to_string(index=False) if not outlier_summary.empty else 'No outliers detected'}

6. RECOMMENDATIONS FOR NEXT PHASE
{'-'*80}
Based on the analysis, the following steps are recommended for Phase 2:

1. Feature Engineering:
   - Calculate geospatial distance between store and drop locations
   - Extract time-based features (hour, day of week, time since order)
   - Create interaction features (e.g., traffic × distance)
   - Encode categorical variables appropriately

2. Feature Selection:
   - Consider correlation analysis results
   - Use feature importance from tree-based models
   - Apply dimensionality reduction if needed

3. Data Transformation:
   - Normalize/standardize numerical features
   - Consider log transformation for skewed features
   - Handle categorical encoding (One-Hot, Label, Target encoding)

4. Model Development:
   - Start with baseline Linear Regression
   - Implement tree-based models (Random Forest, Gradient Boosting)
   - Consider ensemble methods
   - Track experiments using MLflow

{'='*80}
END OF PHASE 1 REPORT
{'='*80}
"""
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Phase 1: Data Analysis & Cleaning')
    parser.add_argument('--data-path', type=str, 
                       default='data/raw/amazon_delivery.csv',
                       help='Path to raw data file')
    
    args = parser.parse_args()
    
    # Run Phase 1 pipeline
    try:
        cleaned_data = run_phase1_pipeline(args.data_path)
        
        if cleaned_data is not None:
            print("\n✅ Pipeline executed successfully!")
            print("\n📊 Next Steps:")
            print("  1. Review the generated reports and visualizations")
            print("  2. Validate the cleaning decisions")
            print("  3. Proceed to Phase 2: Feature Engineering")
            print("\n💡 To proceed to Phase 2, run:")
            print("     python scripts/run_phase2_pipeline.py")
    
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)