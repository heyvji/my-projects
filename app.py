import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

# Page configuration
st.set_page_config(page_title="Streamlit Components Demo", layout="wide")

# Main title
st.title("🚀 Complete Streamlit Components Demo")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a section:", ["Text & Data", "Input Widgets", "Charts", "Layout"])

# Sample data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'London', 'Tokyo', 'Paris'],
    'Salary': [50000, 60000, 70000, 55000]
})

if page == "Text & Data":
    st.header("📝 Text and Data Display")
    
    # st.write() - versatile display function
    st.subheader("st.write() - The Swiss Army Knife")
    st.write("This displays text, dataframes, charts, and more!")
    st.write(df)
    
    # st.markdown() - formatted text
    st.subheader("st.markdown() - Rich Text Formatting")
    st.markdown("""
    **Bold text**, *italic text*, and `code`
    
    - Bullet point 1
    - Bullet point 2
    
    > This is a blockquote
    """)
    
    # st.dataframe() - interactive table
    st.subheader("st.dataframe() - Interactive Data Table")
    st.dataframe(df, use_container_width=True)
    
    # st.table() - static table
    st.subheader("st.table() - Static Table")
    st.table(df.head(2))

elif page == "Input Widgets":
    st.header("🎛️ Input Widgets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # st.button()
        if st.button("Click Me!"):
            st.success("Button clicked!")
        
        # st.text_input()
        name = st.text_input("Enter your name:")
        if name:
            st.write(f"Hello, {name}!")
        
        # st.slider()
        age = st.slider("Select your age:", 0, 100, 25)
        st.write(f"You are {age} years old")
    
    with col2:
        # st.selectbox()
        city = st.selectbox("Choose your city:", ['New York', 'London', 'Tokyo', 'Paris'])
        st.write(f"You selected: {city}")
        
        # st.checkbox()
        agree = st.checkbox("I agree to terms")
        if agree:
            st.write("Thank you for agreeing!")
        
        # Additional widgets
        rating = st.radio("Rate this app:", ["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        st.write(f"Your rating: {rating}")

elif page == "Charts":
    st.header("📊 Interactive Charts")
    
    # Generate sample data for charts
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    # Matplotlib
    st.subheader("Matplotlib Chart")
    fig, ax = plt.subplots()
    ax.plot(chart_data['A'], label='Series A')
    ax.plot(chart_data['B'], label='Series B')
    ax.legend()
    st.pyplot(fig)
    
    # Plotly
    st.subheader("Plotly Interactive Chart")
    fig_plotly = px.scatter(df, x='Age', y='Salary', color='City', 
                           title='Age vs Salary by City')
    st.plotly_chart(fig_plotly, use_container_width=True)
    
    # Altair
    st.subheader("Altair Chart")
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='Age',
        y='Salary',
        color='City',
        tooltip=['Name', 'Age', 'Salary', 'City']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

elif page == "Layout":
    st.header("🏗️ Layout and Structure")
    
    # st.columns()
    st.subheader("st.columns() - Side by Side Layout")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Temperature", "25°C", "2°C")
    with col2:
        st.metric("Humidity", "60%", "-5%")
    with col3:
        st.metric("Pressure", "1013 hPa", "3 hPa")
    
    # st.expander()
    st.subheader("st.expander() - Collapsible Content")
    with st.expander("Click to expand details"):
        st.write("This content is hidden by default!")
        st.dataframe(df)
        st.write("You can put any content inside an expander.")
    
    # Container and empty
    st.subheader("Additional Layout Elements")
    container = st.container()
    with container:
        st.write("This is inside a container")
        st.info("Containers help organize content")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("This demo showcases Streamlit's core components for building interactive web apps!")