import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    return pd.read_csv(filepath)

def basic_info(df):
    print("\n--- Basic Dataset Info ---")
    print("Dataset Shape:", df.shape)
    
    print("\nTarget Distribution:")
    print(df['Target'].value_counts())
    
    print("\nMissing Values per Column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("No missing values found.")

def plot_target_distribution(df):
    print("\nPlotting target distribution...")
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Target', order=df['Target'].value_counts().index)
    plt.title('Target Variable Distribution')
    plt.ylabel('Count')
    plt.xlabel('Target Status')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_correlations(df):
    print("\n--- Correlation Analysis ---")
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df['Target_numeric'] = df['Target'].map(target_mapping)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_with_target = df[numeric_cols].corrwith(df['Target_numeric'])
    
    # Drop self-correlation and sort
    corr_with_target = corr_with_target.drop('Target_numeric').abs().sort_values(ascending=False)
    
    print("Top 10 features correlated with Target:")
    print(corr_with_target.head(10))
    # Clean up the added column
    df.drop(columns=['Target_numeric'], inplace=True)

def plot_feature_distributions(df):
    print("\nPlotting feature distributions by target...")
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Target', y='Age at enrollment')
    plt.title('Age Distribution by Target Status')
    plt.show()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Academic Performance by Target Status', fontsize=16)

    # 1st semester grades
    sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)', ax=axes[0])
    axes[0].set_title('1st Semester Grades')

    # 2nd semester grades  
    sns.boxplot(data=df, x='Target', y='Curricular units 2nd sem (grade)', ax=axes[1])
    axes[1].set_title('2nd Semester Grades')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    df = load_data(r'C:\Users\Thippesh\my projects\medha_tech_intern\dataset.csv')
    basic_info(df)
    analyze_correlations(df)
    plot_target_distribution(df)
    plot_feature_distributions(df)

