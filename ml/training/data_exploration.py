import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Constants aligned with project structure
DATA_PATH = Path('ml/datasets/cyber_threats.csv')
EDA_OUTPUT_DIR = Path('ml/eda_results/')
EDA_OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_analyze():
    """Main function to load data and perform analysis"""
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Basic analysis
    analysis_results = {
        'shape': df.shape,
        'dtypes': str(df.dtypes.to_dict()),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    
    # Save analysis results
    with open(EDA_OUTPUT_DIR/'basic_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate visualizations
    generate_visualizations(df)
    
    return df

def generate_visualizations(df):
    """Generate and save EDA visualizations"""
    # Target distribution
    plt.figure(figsize=(10, 6))
    target_counts = df['attack_type'].value_counts()  # Update column name as needed
    target_counts.plot(kind='bar')
    plt.title('Target Variable Distribution')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR/'target_distribution.png')
    plt.close()
    
    # Numerical features distribution
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(bins=50, figsize=(20, 15))
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR/'numerical_distributions.png')
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(EDA_OUTPUT_DIR/'correlation_matrix.png')
    plt.close()

if __name__ == '__main__':
    df = load_and_analyze()
    print("EDA completed. Results saved to", EDA_OUTPUT_DIR)