import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI

load_dotenv()

def generate_features(user_input):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

    prompt = f"""
    Given a user's task, identify the optimal set of dataset features and metadata necessary for the task.
    The user inputs will be from researchers who are looking for an optimal dataset to train their machine learning model.
    Identify the optimal set of dataset features and metadata that would be necessary to address the task, list at most 5 features, and suggest potential preprocessing steps for the data.  Do not provide an explanation.

    Input:
    Predict weather patterns.
    Output:
    Type of data: Time series
    Dataset features: Temperature, humidity, atmospheric pressure, wind speed, historical weather events.
    Preprocessing steps: Normalize temperature and pressure values, fill missing values using linear interpolation, encode categorical wind speed.

    Input:
    Predict the state of charge for batteries. I want a dataset with open circuit voltage and state of charge as well as other parameters.
    Output:
    Type of data: Time series
    Dataset features: Open circuit voltage, state of charge, temperature, current, voltage, cycles
    Preprocessing steps: Normalize voltage and current, handle outliers in temperature readings, apply a moving average filter to smooth the charge/discharge cycles.

    Now, for the user's task:

    Task: {user_input}
    Type of data and required dataset features (in a comma-separated list with additional metadata), and suggest preprocessing steps.
    """


    # Generate the response
    response = llm.generate([prompt], max_tokens=1000)

    # Extracting the text from the response
    if response.generations:
        generated_text = response.generations[0][0].text
    else:
        generated_text = "No features generated."

    return generated_text

st.title('DataSphere.ai')
user_input = st.text_area("Enter a machine learning task:", "I want to create a machine learning model that predicts...")
if st.button('Generate Features'):
    features = generate_features(user_input)
    st.text_area("Generated Features and Preprocessing Steps", features, height=250)