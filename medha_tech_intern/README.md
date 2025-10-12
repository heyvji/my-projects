# Student Dropout Prediction Streamlit App

A comprehensive machine learning web application for predicting student dropout risk using various academic and demographic factors.

## Features

### 📖 Introduction
- Overview of student dropout prediction
- Model explanation and benefits
- Dataset statistics

### 📊 Data Information
- Dataset overview and statistics
- Feature categories (categorical vs numerical)
- Data quality assessment
- Target variable distribution

### 🔍 Data Exploration
- Feature importance analysis
- Academic performance visualization
- Categorical feature analysis
- Interactive data exploration

### 🤖 Model Training
- Logistic Regression model
- Random Forest model
- Feature importance ranking
- Model comparison

### 📈 Model Performance
- Accuracy metrics
- Confusion matrices
- Precision, Recall, F1-Score
- Detailed performance analysis

### 🎯 Make Prediction
- Interactive prediction form
- Real-time risk assessment
- Personalized recommendations
- Risk interpretation

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the `student dropout.xlsx` file is in the same directory as `streamlit_app.py`
2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```
3. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

## File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── student dropout.xlsx      # Dataset file
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Models Used

### Logistic Regression
- Linear model for binary classification
- Provides probabilistic outputs
- High interpretability
- Good baseline performance

### Random Forest
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance
- Robust to overfitting

## Dataset Features

The model uses 33 features including:

**Academic Factors:**
- Grades (Grade_1, Grade_2, Final_Grade)
- Number of failures
- Study time
- Number of absences

**Demographic Factors:**
- Age, Gender
- Family size and parental status
- Mother and father education levels
- Address type (urban/rural)

**Social Factors:**
- Family support and relationships
- School support
- Internet access
- Extracurricular activities

**Behavioral Factors:**
- Free time and going out frequency
- Alcohol consumption
- Health status
- Travel time to school

## Prediction Output

The app provides:
- Risk probability from both models
- Average risk score
- Risk level classification (Low/Medium/High)
- Personalized recommendations for at-risk students

## Risk Levels

- 🟢 **Low Risk (< 30%)**: Student likely to complete studies
- 🟡 **Medium Risk (30-70%)**: May benefit from additional support
- 🔴 **High Risk (> 70%)**: Immediate intervention recommended

## Technical Details

- Built with Streamlit for interactive web interface
- Uses scikit-learn for machine learning models
- Pandas for data manipulation
- Matplotlib and Seaborn for visualizations
- Responsive design with custom CSS styling

## Contributing

Feel free to contribute by:
- Adding new features
- Improving model performance
- Enhancing visualizations
- Fixing bugs or issues

## License

This project is for educational purposes.