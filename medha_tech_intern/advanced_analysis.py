import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the dataset from Excel file"""
    return pd.read_excel(file_path)

def feature_importance_analysis(df):
    """Analyze feature importance using Random Forest"""
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Prepare data for machine learning
    df_ml = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
    
    # Separate features and target
    X = df_ml.drop('Dropped_Out', axis=1)
    y = df_ml['Dropped_Out'].astype(int)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance for Dropout Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def grade_progression_analysis(df):
    """Analyze grade progression patterns"""
    print("\n" + "=" * 60)
    print("GRADE PROGRESSION ANALYSIS")
    print("=" * 60)
    
    # Calculate grade changes
    df['Grade_1_to_2_Change'] = df['Grade_2'] - df['Grade_1']
    df['Grade_2_to_Final_Change'] = df['Final_Grade'] - df['Grade_2']
    df['Overall_Grade_Change'] = df['Final_Grade'] - df['Grade_1']
    
    # Analyze grade changes by dropout status
    print("Grade Change Analysis:")
    grade_changes = ['Grade_1_to_2_Change', 'Grade_2_to_Final_Change', 'Overall_Grade_Change']
    
    for change in grade_changes:
        dropped_mean = df[df['Dropped_Out'] == True][change].mean()
        continued_mean = df[df['Dropped_Out'] == False][change].mean()
        print(f"{change}:")
        print(f"  Dropped Out: {dropped_mean:.2f}")
        print(f"  Continued: {continued_mean:.2f}")
        print()
    
    # Visualize grade progression
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Grade progression for dropouts vs non-dropouts
    grades = ['Grade_1', 'Grade_2', 'Final_Grade']
    
    # Line plot for average grades
    dropped_grades = df[df['Dropped_Out'] == True][grades].mean()
    continued_grades = df[df['Dropped_Out'] == False][grades].mean()
    
    axes[0,0].plot(grades, dropped_grades, marker='o', label='Dropped Out', color='red')
    axes[0,0].plot(grades, continued_grades, marker='o', label='Continued', color='blue')
    axes[0,0].set_title('Average Grade Progression')
    axes[0,0].set_ylabel('Average Grade')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Distribution of grade changes
    for i, change in enumerate(grade_changes):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        df[df['Dropped_Out'] == False][change].hist(alpha=0.7, label='Continued', 
                                                   bins=20, ax=axes[row, col])
        df[df['Dropped_Out'] == True][change].hist(alpha=0.7, label='Dropped Out', 
                                                  bins=20, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {change}')
        axes[row, col].set_xlabel('Grade Change')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.show()

def risk_factor_analysis(df):
    """Analyze risk factors for dropout"""
    print("\n" + "=" * 60)
    print("RISK FACTOR ANALYSIS")
    print("=" * 60)
    
    # Define risk categories
    risk_factors = {
        'Academic_Risk': ['Number_of_Failures', 'Final_Grade', 'Study_Time'],
        'Social_Risk': ['Going_Out', 'In_Relationship', 'Weekend_Alcohol_Consumption'],
        'Family_Risk': ['Family_Support', 'Family_Relationship', 'Parental_Status'],
        'School_Risk': ['School_Support', 'Number_of_Absences', 'Travel_Time']
    }
    
    # Calculate risk scores
    df_risk = df.copy()
    
    # Normalize numerical features to 0-1 scale for risk calculation
    numerical_cols = df_risk.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['Dropped_Out']:
            df_risk[col] = (df_risk[col] - df_risk[col].min()) / (df_risk[col].max() - df_risk[col].min())
    
    # Calculate composite risk scores
    for risk_category, features in risk_factors.items():
        available_features = [f for f in features if f in df_risk.columns]
        if available_features:
            # For academic risk, lower grades mean higher risk
            if risk_category == 'Academic_Risk':
                df_risk[risk_category] = df_risk[available_features].apply(
                    lambda x: np.mean([1-x[f] if f in ['Final_Grade', 'Study_Time'] else x[f] 
                                     for f in available_features]), axis=1)
            else:
                df_risk[risk_category] = df_risk[available_features].mean(axis=1)
    
    # Analyze risk scores by dropout status
    risk_categories = list(risk_factors.keys())
    
    print("Risk Score Analysis:")
    for category in risk_categories:
        if category in df_risk.columns:
            dropped_risk = df_risk[df_risk['Dropped_Out'] == True][category].mean()
            continued_risk = df_risk[df_risk['Dropped_Out'] == False][category].mean()
            print(f"{category}:")
            print(f"  Dropped Out: {dropped_risk:.3f}")
            print(f"  Continued: {continued_risk:.3f}")
            print()
    
    # Visualize risk factors
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, category in enumerate(risk_categories):
        if category in df_risk.columns and i < len(axes):
            df_risk.boxplot(column=category, by='Dropped_Out', ax=axes[i])
            axes[i].set_title(f'{category} by Dropout Status')
            axes[i].set_xlabel('Dropped Out')
            axes[i].set_ylabel('Risk Score')
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.show()

def predictive_model_analysis(df):
    """Build and evaluate a predictive model"""
    print("\n" + "=" * 60)
    print("PREDICTIVE MODEL ANALYSIS")
    print("=" * 60)
    
    # Prepare data
    df_ml = df.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_ml[col] = le.fit_transform(df_ml[col])
    
    # Separate features and target
    X = df_ml.drop('Dropped_Out', axis=1)
    y = df_ml['Dropped_Out'].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate model
    print("Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Continued', 'Dropped Out']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Continued', 'Dropped Out'],
                yticklabels=['Continued', 'Dropped Out'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return rf

def generate_recommendations(df, feature_importance):
    """Generate actionable recommendations based on analysis"""
    print("\n" + "=" * 60)
    print("ACTIONABLE RECOMMENDATIONS")
    print("=" * 60)
    
    # Get top risk factors
    top_factors = feature_importance.head(10)['feature'].tolist()
    
    recommendations = {
        'Final_Grade': "Implement early warning systems for students with declining grades",
        'Grade_2': "Provide additional support after second-period assessments",
        'Grade_1': "Identify at-risk students early in the academic year",
        'Number_of_Failures': "Offer remedial classes and tutoring for students with failures",
        'Study_Time': "Promote effective study habits and time management skills",
        'Father_Education': "Engage parents in educational support programs",
        'Mother_Education': "Provide family literacy programs",
        'School': "MS school needs targeted intervention programs",
        'Age': "Monitor older students who may be at higher risk",
        'Weekend_Alcohol_Consumption': "Implement substance abuse prevention programs"
    }
    
    print("Priority Interventions (based on feature importance):")
    for i, factor in enumerate(top_factors[:5], 1):
        if factor in recommendations:
            print(f"{i}. {factor}: {recommendations[factor]}")
    
    # School-specific recommendations
    print("\nSchool-Specific Insights:")
    school_dropout = df.groupby('School')['Dropped_Out'].mean() * 100
    for school, rate in school_dropout.items():
        print(f"- {school} School: {rate:.1f}% dropout rate")
        if rate > 20:
            print("  * HIGH PRIORITY: Needs immediate intervention")
        elif rate > 15:
            print("  * MEDIUM PRIORITY: Monitor closely")
        else:
            print("  * LOW PRIORITY: Maintain current practices")
    
    # Gender-specific recommendations
    print("\nGender-Specific Insights:")
    gender_dropout = df.groupby('Gender')['Dropped_Out'].mean() * 100
    for gender, rate in gender_dropout.items():
        print(f"- {gender}: {rate:.1f}% dropout rate")
        if gender == 'M' and rate > gender_dropout['F']:
            print("  * Focus on male student engagement programs")

if __name__ == "__main__":
    # Load the dataset
    df = load_data(r'C:\Users\Thippesh\my projects\medha_tech_intern\student dropout.xlsx')
    
    # Perform advanced analysis
    feature_importance = feature_importance_analysis(df)
    grade_progression_analysis(df)
    risk_factor_analysis(df)
    model = predictive_model_analysis(df)
    generate_recommendations(df, feature_importance)
    
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("=" * 60)