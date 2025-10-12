import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration/layout
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Design of the pages/sections
st.markdown("""
<style>
body {
    color: #FAFAFA;
    background-color: #1E1E1E;
}
.stApp {
    background-color: #1E1E1E;
}
.main-header {
    font-size: 3rem;
    color: #4FC3F7;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 2rem;
    color: #FFA726;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #2C2C2C;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid #444;
}
/* Sidebar style */
.css-1d391kg {
    background-color: #2C2C2C;
}
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('student dropout.xlsx')
        return df
    except FileNotFoundError:
        st.error("Please ensure 'student dropout.xlsx' is in the same directory as this app.")
        return None

# Data preprocessing
@st.cache_data
def preprocess_data(df):
    X = df.drop('Dropped_Out', axis=1)
    y = df['Dropped_Out']
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    X_processed = X.copy()
    label_encoders = {}
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X_processed[col])
        label_encoders[col] = le
    
    # Feature engineering
    X_processed['Average_Grade'] = (X_processed['Grade_1'] + X_processed['Grade_2'] + X_processed['Final_Grade']) / 3
    X_processed['Grade_Improvement'] = X_processed['Final_Grade'] - X_processed['Grade_1']
    X_processed['Study_Free_Ratio'] = X_processed['Study_Time'] / (X_processed['Free_Time'] + 1)
    X_processed['Total_Alcohol'] = X_processed['Weekend_Alcohol_Consumption'] + X_processed['Weekday_Alcohol_Consumption']
    
    return X_processed, y, categorical_cols, numerical_cols, label_encoders

# Train models
@st.cache_data
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    return {
        'models': {'Logistic Regression': lr_model, 'Random Forest': rf_model},
        'scaler': scaler,
        'test_data': (X_test, X_test_scaled, y_test),
        'predictions': {'Logistic Regression': lr_pred, 'Random Forest': rf_pred},
        'accuracies': {'Logistic Regression': lr_accuracy, 'Random Forest': rf_accuracy}
    }

def main():
    st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    sections = [
        "📖 Introduction",
        "📊 Data Information", 
        "🔍 Data Exploration",
        "🤖 Model Training",
        "📈 Model Performance",
        "🎯 Make Prediction"
    ]
    
    selected_section = st.sidebar.selectbox("Choose a section:", sections)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    X_processed, y, categorical_cols, numerical_cols, label_encoders = preprocess_data(df)
    
    # Introduction Section
    if selected_section == "📖 Introduction":
        st.markdown('<h2 class="section-header">About This Model</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### What is Student Dropout Prediction?
            
            Student dropout prediction is a machine learning application that helps educational institutions identify students who are at risk of dropping out before completing their studies. This early warning system enables:
            
            - **Early Intervention**: Identify at-risk students before they drop out
            - **Resource Allocation**: Focus support resources on students who need them most  
            - **Improved Retention**: Increase graduation rates through targeted interventions
            - **Data-driven Decisions**: Make informed decisions based on student data patterns
            
            ### How It Works
            
            Our model analyzes various student factors including:
            - **Academic Performance**: Grades, study time, previous failures
            - **Demographic Information**: Age, family background, parental education
            - **Social Factors**: Family support, relationships, extracurricular activities
            - **Behavioral Patterns**: Attendance, alcohol consumption, free time activities
            
            The system uses machine learning algorithms to identify patterns in historical data and predict the likelihood of a student dropping out.
            """)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>📊 Dataset Overview</h3>
            <ul>
            <li><strong>Total Students:</strong> 649</li>
            <li><strong>Features:</strong> 33</li>
            <li><strong>Dropout Rate:</strong> 15.4%</li>
            <li><strong>Completion Rate:</strong> 84.6%</li>
            </ul>
            </div>
            
            <div class="metric-card">
            <h3>🎯 Model Benefits</h3>
            <ul>
            <li>Early risk detection</li>
            <li>Personalized interventions</li>
            <li>Improved student outcomes</li>
            <li>Cost-effective resource use</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Data Information Section
    elif selected_section == "📊 Data Information":
        st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            st.metric("Dropout Cases", df['Dropped_Out'].sum())
        with col4:
            st.metric("Completion Cases", len(df) - df['Dropped_Out'].sum())
        
        st.subheader("📋 Feature Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Categorical Features:**")
            for col in categorical_cols:
                st.write(f"• {col}")
        
        with col2:
            st.markdown("**Numerical Features:**")
            for col in numerical_cols:
                st.write(f"• {col}")
        
        st.subheader("🔍 Data Quality")
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.dataframe(missing_data[missing_data > 0])
        
        st.subheader("📊 Target Variable Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        dropout_counts = df['Dropped_Out'].value_counts()
        ax[0].pie(dropout_counts.values, labels=['Completed', 'Dropped Out'], autopct='%1.1f%%', startangle=90)
        ax[0].set_title('Dropout Distribution')
        
        # Bar plot
        ax[1].bar(['Completed', 'Dropped Out'], dropout_counts.values, color=['green', 'red'])
        ax[1].set_title('Student Counts')
        ax[1].set_ylabel('Number of Students')
        
        st.pyplot(fig)
    
    # Data Exploration Section
    elif selected_section == "🔍 Data Exploration":
        st.markdown('<h2 class="section-header">Data Exploration & Analysis</h2>', unsafe_allow_html=True)
        
        # Feature correlation with target
        st.subheader("🎯 Feature Importance Analysis")
        
        correlation_data = X_processed.copy()
        correlation_data['Dropped_Out'] = y.astype(int)
        correlations = correlation_data.corr()['Dropped_Out'].abs().sort_values(ascending=False)[1:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = correlations.head(15)
        ax.barh(range(len(top_features)), top_features.values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Correlation with Dropout')
        ax.set_title('Top 15 Features Correlated with Dropout')
        st.pyplot(fig)
        
        # Grade analysis
        st.subheader("📚 Academic Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            for grade_col in ['Grade_1', 'Grade_2', 'Final_Grade']:
                df_dropped = df[df['Dropped_Out'] == True][grade_col]
                df_completed = df[df['Dropped_Out'] == False][grade_col]
                
                ax.hist(df_completed, alpha=0.7, label=f'{grade_col} - Completed', bins=20)
                ax.hist(df_dropped, alpha=0.7, label=f'{grade_col} - Dropped', bins=20)
            
            ax.set_xlabel('Grade')
            ax.set_ylabel('Frequency')
            ax.set_title('Grade Distribution by Dropout Status')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            # Average grades comparison
            avg_grades_completed = df[df['Dropped_Out'] == False][['Grade_1', 'Grade_2', 'Final_Grade']].mean()
            avg_grades_dropped = df[df['Dropped_Out'] == True][['Grade_1', 'Grade_2', 'Final_Grade']].mean()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.arange(len(avg_grades_completed))
            width = 0.35
            
            ax.bar(x - width/2, avg_grades_completed, width, label='Completed', color='green', alpha=0.7)
            ax.bar(x + width/2, avg_grades_dropped, width, label='Dropped Out', color='red', alpha=0.7)
            
            ax.set_xlabel('Grade Type')
            ax.set_ylabel('Average Grade')
            ax.set_title('Average Grades Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(['Grade 1', 'Grade 2', 'Final Grade'])
            ax.legend()
            st.pyplot(fig)
        
        # Categorical feature analysis
        st.subheader("📊 Categorical Features Analysis")
        
        selected_cat_feature = st.selectbox("Select a categorical feature to analyze:", categorical_cols)
        
        if selected_cat_feature:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create crosstab
            crosstab = pd.crosstab(df[selected_cat_feature], df['Dropped_Out'], normalize='index') * 100
            crosstab.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
            
            ax.set_title(f'Dropout Rate by {selected_cat_feature}')
            ax.set_xlabel(selected_cat_feature)
            ax.set_ylabel('Percentage')
            ax.legend(['Completed', 'Dropped Out'])
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    # Model Training Section
    elif selected_section == "🤖 Model Training":
        st.markdown('<h2 class="section-header">Machine Learning Models</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Models Used
        
        We employ two different machine learning algorithms to predict student dropout:
        
        1. **Logistic Regression**: A linear model that's interpretable and works well for binary classification
        2. **Random Forest**: An ensemble method that combines multiple decision trees for better accuracy
        """)
        
        # Train models
        with st.spinner("Training models..."):
            results = train_models(X_processed, y)
        
        st.success("✅ Models trained successfully!")
        
        # Display model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Logistic Regression**
            - Linear relationship modeling
            - Probabilistic output
            - High interpretability
            - Good baseline performance
            """)
        
        with col2:
            st.markdown("""
            **🌳 Random Forest**
            - Ensemble of decision trees
            - Handles non-linear relationships
            - Feature importance ranking
            - Robust to overfitting
            """)
        
        # Feature importance for Random Forest
        st.subheader("🎯 Feature Importance (Random Forest)")
        
        rf_model = results['models']['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(15)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 15 Most Important Features')
        st.pyplot(fig)
    
    # Model Performance Section
    elif selected_section == "📈 Model Performance":
        st.markdown('<h2 class="section-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
        
        # Train models
        results = train_models(X_processed, y)
        
        # Display accuracies
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Logistic Regression Accuracy", f"{results['accuracies']['Logistic Regression']:.3f}")
        
        with col2:
            st.metric("Random Forest Accuracy", f"{results['accuracies']['Random Forest']:.3f}")
        
        # Confusion matrices
        st.subheader("🔍 Confusion Matrices")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (model_name, predictions) in enumerate(results['predictions'].items()):
            cm = confusion_matrix(results['test_data'][2], predictions)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{model_name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        st.pyplot(fig)
        
        # Classification reports
        st.subheader("📊 Detailed Performance Metrics")
        
        for model_name, predictions in results['predictions'].items():
            st.markdown(f"**{model_name}:**")
            report = classification_report(results['test_data'][2], predictions, output_dict=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision (Dropout)", f"{report['True']['precision']:.3f}")
            with col2:
                st.metric("Recall (Dropout)", f"{report['True']['recall']:.3f}")
            with col3:
                st.metric("F1-Score (Dropout)", f"{report['True']['f1-score']:.3f}")
    
    # Prediction Section
    elif selected_section == "🎯 Make Prediction":
        st.markdown('<h2 class="section-header">Student Dropout Prediction</h2>', unsafe_allow_html=True)
        
        st.markdown("Enter student information to predict dropout risk:")
        
        # Train models
        results = train_models(X_processed, y)
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            # Get sample data for reference
            sample_student = df.iloc[0]
            
            with col1:
                st.subheader("📚 Academic Info")
                age = st.number_input("Age", min_value=15, max_value=25, value=18)
                grade_1 = st.number_input("Grade 1", min_value=0, max_value=20, value=12)
                grade_2 = st.number_input("Grade 2", min_value=0, max_value=20, value=12)
                final_grade = st.number_input("Final Grade", min_value=0, max_value=20, value=12)
                failures = st.number_input("Number of Failures", min_value=0, max_value=5, value=0)
                absences = st.number_input("Number of Absences", min_value=0, max_value=50, value=5)
                
            with col2:
                st.subheader("👨‍👩‍👧‍👦 Family Info")
                mother_edu = st.number_input("Mother Education Level", min_value=0, max_value=4, value=2)
                father_edu = st.number_input("Father Education Level", min_value=0, max_value=4, value=2)
                family_rel = st.number_input("Family Relationship Quality", min_value=1, max_value=5, value=4)
                
                school_support = st.selectbox("School Support", ["yes", "no"])
                family_support = st.selectbox("Family Support", ["yes", "no"])
                higher_edu = st.selectbox("Wants Higher Education", ["yes", "no"])
                
            with col3:
                st.subheader("⏰ Time & Activities")
                study_time = st.number_input("Study Time", min_value=1, max_value=4, value=2)
                free_time = st.number_input("Free Time", min_value=1, max_value=5, value=3)
                going_out = st.number_input("Going Out Frequency", min_value=1, max_value=5, value=3)
                
                travel_time = st.number_input("Travel Time to School", min_value=1, max_value=4, value=1)
                health = st.number_input("Health Status", min_value=1, max_value=5, value=5)
                
                internet = st.selectbox("Internet Access", ["yes", "no"])
                
            submitted = st.form_submit_button("🔮 Predict Dropout Risk")
            
            if submitted:
                # Create input data
                input_data = {
                    'School': 0,  # Default values for required fields
                    'Gender': 0,
                    'Age': age,
                    'Address': 0,
                    'Family_Size': 0,
                    'Parental_Status': 0,
                    'Mother_Education': mother_edu,
                    'Father_Education': father_edu,
                    'Mother_Job': 0,
                    'Father_Job': 0,
                    'Reason_for_Choosing_School': 0,
                    'Guardian': 0,
                    'Travel_Time': travel_time,
                    'Study_Time': study_time,
                    'Number_of_Failures': failures,
                    'School_Support': 1 if school_support == "yes" else 0,
                    'Family_Support': 1 if family_support == "yes" else 0,
                    'Extra_Paid_Class': 0,
                    'Extra_Curricular_Activities': 0,
                    'Attended_Nursery': 0,
                    'Wants_Higher_Education': 1 if higher_edu == "yes" else 0,
                    'Internet_Access': 1 if internet == "yes" else 0,
                    'In_Relationship': 0,
                    'Family_Relationship': family_rel,
                    'Free_Time': free_time,
                    'Going_Out': going_out,
                    'Weekend_Alcohol_Consumption': 1,
                    'Weekday_Alcohol_Consumption': 1,
                    'Health_Status': health,
                    'Number_of_Absences': absences,
                    'Grade_1': grade_1,
                    'Grade_2': grade_2,
                    'Final_Grade': final_grade
                }
                
                # Add engineered features
                input_data['Average_Grade'] = (grade_1 + grade_2 + final_grade) / 3
                input_data['Grade_Improvement'] = final_grade - grade_1
                input_data['Study_Free_Ratio'] = study_time / (free_time + 1)
                input_data['Total_Alcohol'] = 2  # Default value
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make predictions
                lr_model = results['models']['Logistic Regression']
                rf_model = results['models']['Random Forest']
                scaler = results['scaler']
                
                # Scale input for logistic regression
                input_scaled = scaler.transform(input_df)
                
                # Get predictions
                lr_prob = lr_model.predict_proba(input_scaled)[0][1]
                rf_prob = rf_model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Logistic Regression Risk", f"{lr_prob:.1%}")
                    if lr_prob > 0.5:
                        st.error("⚠️ High dropout risk detected!")
                    else:
                        st.success("✅ Low dropout risk")
                
                with col2:
                    st.metric("Random Forest Risk", f"{rf_prob:.1%}")
                    if rf_prob > 0.5:
                        st.error("⚠️ High dropout risk detected!")
                    else:
                        st.success("✅ Low dropout risk")
                
                # Average risk
                avg_risk = (lr_prob + rf_prob) / 2
                st.markdown(f"**Average Risk Score: {avg_risk:.1%}**")
                
                # Risk interpretation
                if avg_risk < 0.3:
                    st.success("🟢 **Low Risk**: Student is likely to complete their studies successfully.")
                elif avg_risk < 0.7:
                    st.warning("🟡 **Medium Risk**: Student may benefit from additional support and monitoring.")
                else:
                    st.error("🔴 **High Risk**: Immediate intervention recommended to prevent dropout.")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if avg_risk > 0.5:
                    recommendations = []
                    
                    if grade_1 < 10 or grade_2 < 10 or final_grade < 10:
                        recommendations.append("📚 Provide academic tutoring and study support")
                    
                    if failures > 0:
                        recommendations.append("🎯 Implement personalized learning plan")
                    
                    if family_support == "no":
                        recommendations.append("👨‍👩‍👧‍👦 Engage family in student's education")
                    
                    if study_time < 2:
                        recommendations.append("⏰ Encourage increased study time")
                    
                    if absences > 10:
                        recommendations.append("📅 Address attendance issues")
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.write("• Continue current support level")
                    st.write("• Monitor progress regularly")
                    st.write("• Maintain positive learning environment")

if __name__ == "__main__":
    main()