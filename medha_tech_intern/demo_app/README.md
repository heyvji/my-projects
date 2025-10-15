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

## Docker (Containerized) Usage

Make sure Docker is installed and running on your machine. Also ensure the dataset file `student dropout.xlsx` is present in the project folder (it is ignored by the image via `.dockerignore`; mounting or copying it into the container is required).

Build the Docker image (PowerShell):
```powershell
docker build -t student-dropout-app .
```

Run the container exposing Streamlit port:
```powershell
docker run --rm -p 8501:8501 -v ${PWD}: /app student-dropout-app
```

Or use docker-compose (PowerShell):
```powershell
docker-compose up --build
```

Open http://localhost:8501 in your browser.

Notes:
- If you mount the project directory into the container (with `-v ${PWD}:/app` or via `docker-compose`), the container will be able to read the Excel dataset placed in the host folder.
- If the app errors with FileNotFoundError, double-check the dataset filename and that it exists in the project root.

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

## Future enhancements

- Adding new features
- Improving model performance
- using different model
- Enhancing visualizations
- Fixing bugs or issues

## Use case

This project is for educational purposes.