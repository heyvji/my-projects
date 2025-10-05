import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_data(file_path):
    return pd.read_excel(file_path)

def basic_info(df):
    print("DATASET OVERVIEW")
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Features: {df.shape[1]}")
    print(f"Number of Records: {df.shape[0]}")
    print(f"Column Names: {df.columns.tolist()}")
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
def missing_values_analysis(df):
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("[OK] No missing values found in the dataset!")
    else:
        missing_percent = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])

def target_variable_analysis(df):
    
    dropout_counts = df['Dropped_Out'].value_counts()
    dropout_percent = df['Dropped_Out'].value_counts(normalize=True) * 100
    
    print("Dropout Distribution:")
    for i, (status, count) in enumerate(dropout_counts.items()):
        print(f"  {'Dropped Out' if status else 'Continued'}: {count} ({dropout_percent.iloc[i]:.1f}%)")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    dropout_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
    ax1.set_title('Student Dropout Distribution')
    ax1.set_xlabel('Status')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Continued', 'Dropped Out'], rotation=0)
    
    # Pie chart
    ax2.pie(dropout_counts.values, labels=['Continued', 'Dropped Out'], 
            autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    ax2.set_title('Dropout Rate Distribution')
    
    plt.tight_layout()
    plt.show()

def numerical_features_analysis(df):
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Dropped_Out' in numerical_cols:
        numerical_cols.remove('Dropped_Out')
    
    print(f"Number of numerical features: {len(numerical_cols)}")
    print("\nNumerical Features:")
    for col in numerical_cols:
        print(f"  - {col}")
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df[numerical_cols].describe())
    
    # Distribution plots
    n_cols = 4
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            df[col].hist(bins=20, ax=axes[i], alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def categorical_features_analysis(df):
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Number of categorical features: {len(categorical_cols)}")
    print("\nCategorical Features:")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"  - {col}: {unique_count} unique values")
    
    # Value counts for each categorical feature
    print("\nValue Counts for Categorical Features:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Visualization for key categorical features
    key_features = ['Gender', 'School', 'Address', 'Family_Size', 'Parental_Status']
    available_features = [col for col in key_features if col in categorical_cols]
    
    if available_features:
        n_cols = 3
        n_rows = (len(available_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(available_features):
            if i < len(axes):
                df[col].value_counts().plot(kind='bar', ax=axes[i], color='lightcoral')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def correlation_analysis(df):
    # Convert boolean to numeric for correlation
    df_corr = df.copy()
    df_corr['Dropped_Out'] = df_corr['Dropped_Out'].astype(int)
    
    numerical_cols = df_corr.select_dtypes(include=[np.number]).columns
    corr_matrix = df_corr[numerical_cols].corr()
    
    # Correlation with target variable
    target_corr = corr_matrix['Dropped_Out'].sort_values(key=abs, ascending=False)
    print("Correlation with Dropout Status:")
    for feature, corr_val in target_corr.items():
        if feature != 'Dropped_Out':
            print(f"  {feature}: {corr_val:.3f}")
    
    # Correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def dropout_analysis_by_features(df):
    categorical_cols = ['Gender', 'School', 'Address', 'Family_Size']
    available_cats = [col for col in categorical_cols if col in df.columns]
    
    for col in available_cats:
        print(f"\nDropout Rate by {col}:")
        dropout_by_feature = df.groupby(col)['Dropped_Out'].agg(['count', 'sum', 'mean'])
        dropout_by_feature.columns = ['Total', 'Dropped_Out', 'Dropout_Rate']
        dropout_by_feature['Dropout_Rate'] = dropout_by_feature['Dropout_Rate'] * 100
        print(dropout_by_feature)
    
    # Visualization
    if available_cats:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(available_cats[:4]):
            if i < len(axes):
                dropout_rates = df.groupby(col)['Dropped_Out'].mean() * 100
                dropout_rates.plot(kind='bar', ax=axes[i], color='lightsteelblue')
                axes[i].set_title(f'Dropout Rate by {col}')
                axes[i].set_ylabel('Dropout Rate (%)')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def grade_analysis(df):
    
    grade_cols = ['Grade_1', 'Grade_2', 'Final_Grade']
    available_grades = [col for col in grade_cols if col in df.columns]
    
    if available_grades:
        print("Grade Statistics:")
        print(df[available_grades].describe())
        
        # Grade comparison between dropouts and non-dropouts
        print("\nGrade Comparison (Dropout vs Non-Dropout):")
        for col in available_grades:
            dropped = df[df['Dropped_Out'] == True][col].mean()
            continued = df[df['Dropped_Out'] == False][col].mean()
            print(f"{col}: Dropped Out = {dropped:.2f}, Continued = {continued:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, col in enumerate(available_grades):
            df.boxplot(column=col, by='Dropped_Out', ax=axes[i])
            axes[i].set_title(f'{col} by Dropout Status')
            axes[i].set_xlabel('Dropped Out')
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.show()

def key_insights(df):
    """Generate key insights from the analysis"""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Basic statistics
    total_students = len(df)
    dropout_count = df['Dropped_Out'].sum()
    dropout_rate = (dropout_count / total_students) * 100
    
    print(f"Dataset contains {total_students} student records")
    print(f"Overall dropout rate: {dropout_rate:.1f}% ({dropout_count} students)")
    
    # Grade insights
    if 'Final_Grade' in df.columns:
        avg_grade_dropped = df[df['Dropped_Out'] == True]['Final_Grade'].mean()
        avg_grade_continued = df[df['Dropped_Out'] == False]['Final_Grade'].mean()
        print(f"Average final grade - Dropped: {avg_grade_dropped:.1f}, Continued: {avg_grade_continued:.1f}")
    
    # Gender insights
    if 'Gender' in df.columns:
        gender_dropout = df.groupby('Gender')['Dropped_Out'].mean() * 100
        for gender, rate in gender_dropout.items():
            print(f"{gender} dropout rate: {rate:.1f}%")
    
    # Failure insights
    if 'Number_of_Failures' in df.columns:
        avg_failures_dropped = df[df['Dropped_Out'] == True]['Number_of_Failures'].mean()
        avg_failures_continued = df[df['Dropped_Out'] == False]['Number_of_Failures'].mean()
        print(f"Average failures - Dropped: {avg_failures_dropped:.1f}, Continued: {avg_failures_continued:.1f}")

if __name__ == "__main__":
    
    df = load_data(r'C:\Users\Thippesh\my projects\medha_tech_intern\student dropout.xlsx')
    
    basic_info(df)
    # missing_values_analysis(df)
    target_variable_analysis(df)
    numerical_features_analysis(df)
    categorical_features_analysis(df)
    correlation_analysis(df)
dropout_analysis_by_features(df)
key_insights(df)
    