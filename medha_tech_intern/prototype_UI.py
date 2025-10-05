import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Student Dropout Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_excel(r'C:\Users\Thippesh\my projects\medha_tech_intern\student dropout.xlsx')

def main():
    st.title("📊 Student Dropout Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Dataset Overview", "Target Analysis", "Numerical Features", "Categorical Features", "Correlation Analysis", "Dropout by Features"]
    )
    
    if analysis_type == "Dataset Overview":
        dataset_overview(df)
    elif analysis_type == "Target Analysis":
        target_analysis(df)
    elif analysis_type == "Numerical Features":
        numerical_analysis(df)
    elif analysis_type == "Categorical Features":
        categorical_analysis(df)
    elif analysis_type == "Correlation Analysis":
        correlation_analysis(df)
    elif analysis_type == "Dropout by Features":
        dropout_by_features(df)

def dataset_overview(df):
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Dropout Rate", f"{(df['Dropped_Out'].sum()/len(df)*100):.1f}%")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("Data Types")
    st.write(df.dtypes.value_counts())
    
    st.subheader("Sample Data")
    st.dataframe(df.head())

def target_analysis(df):
    st.header("🎯 Target Variable Analysis")
    
    dropout_counts = df['Dropped_Out'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=['Continued', 'Dropped Out'],
            y=[dropout_counts[False], dropout_counts[True]],
            title="Student Dropout Distribution",
            color=['Continued', 'Dropped Out'],
            color_discrete_map={'Continued': 'skyblue', 'Dropped Out': 'salmon'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=dropout_counts.values,
            names=['Continued', 'Dropped Out'],
            title="Dropout Rate Distribution",
            color_discrete_map={'Continued': 'skyblue', 'Dropped Out': 'salmon'}
        )
        st.plotly_chart(fig, use_container_width=True)

def numerical_analysis(df):
    st.header("🔢 Numerical Features Analysis")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Dropped_Out' in numerical_cols:
        numerical_cols.remove('Dropped_Out')
    
    st.subheader("Statistical Summary")
    st.dataframe(df[numerical_cols].describe())
    
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select Feature", numerical_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=selected_feature, title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_feature, title=f"Box Plot of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

def categorical_analysis(df):
    st.header("📊 Categorical Features Analysis")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        selected_feature = st.selectbox("Select Categorical Feature", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            value_counts = df[selected_feature].value_counts()
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Distribution of {selected_feature}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Value Counts")
            st.write(value_counts)

def correlation_analysis(df):
    st.header("🔗 Correlation Analysis")
    
    df_corr = df.copy()
    df_corr['Dropped_Out'] = df_corr['Dropped_Out'].astype(int)
    
    numerical_cols = df_corr.select_dtypes(include=[np.number]).columns
    corr_matrix = df_corr[numerical_cols].corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Correlation with Dropout")
        target_corr = corr_matrix['Dropped_Out'].sort_values(key=abs, ascending=False)
        for feature, corr_val in target_corr.items():
            if feature != 'Dropped_Out':
                st.write(f"**{feature}**: {corr_val:.3f}")

def dropout_by_features(df):
    st.header("📈 Dropout Analysis by Features")
    
    categorical_cols = ['Gender', 'Address', 'Family_Size', 'In_re']
    available_cats = [col for col in categorical_cols if col in df.columns]
    
    if available_cats:
        selected_feature = st.selectbox("Select Feature", available_cats)
        
        dropout_by_feature = df.groupby(selected_feature)['Dropped_Out'].agg(['count', 'sum', 'mean'])
        dropout_by_feature.columns = ['Total', 'Dropped_Out', 'Dropout_Rate']
        dropout_by_feature['Dropout_Rate'] = dropout_by_feature['Dropout_Rate'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=dropout_by_feature.index,
                y=dropout_by_feature['Dropout_Rate'],
                title=f"Dropout Rate by {selected_feature}",
                labels={'y': 'Dropout Rate (%)', 'x': selected_feature}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Detailed Statistics")
            st.dataframe(dropout_by_feature)

if __name__ == "__main__":
    main()