import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
import pandas as pd
import json
from similarity import Similarity
from langchain_experimental.agents import create_csv_agent

json_file_path = '/Users/vinithamarupeddi/Desktop/catapult_hacks/catapulthacks/metadata.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

load_dotenv()

def generate_features(user_input):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

    prompt = f"""
    Given a user's task, identify the optimal set of dataset features and metadata necessary for the task.
    The user inputs will be from researchers who are looking for an optimal dataset to train their machine learning model.
    Identify the optimal set of tabular dataset features and metadata that would be necessary to address the task, list at most 5 features, and suggest potential preprocessing steps for the data.  Do not provide an explanation.
    Also generate a potential title for the dataset.
    Input:
    I want a dataset that can be used to predict weather patterns.
    Output:
    Title: Historical Weather Patterns
    Dataset features: Temperature, humidity, atmospheric pressure, wind speed, historical weather events.

    Input:
    I want to predict the state of charge for batteries. I want a dataset with open circuit voltage and state of charge as well as other parameters.
    Output:
    Title: State of Charge Batteries Dataset
    Dataset features: Open circuit voltage, state of charge, temperature, current, voltage, cycles

    Now, for the user's task:

    Task: {user_input}
    Type of data and required dataset features (in a comma-separated list)
    """


    # Generate the response
    response = llm.generate([prompt], max_tokens=1000)

    # Extracting the text from the response
    if response.generations:
        generated_text = response.generations[0][0].text
    else:
        generated_text = "No features generated."

    print(generated_text)

    lines = generated_text.strip().split('\n')
    # Initialize empty strings for title and features
    title = ""
    dataset_features = ""

    # Loop through each line and parse out the title and features
    for line in lines:
        # Check if the line starts with 'Title:'
        if line.startswith('Title:'):
            title = line.split(':', 1)[1].strip()
        # Check if the line starts with 'Dataset features:'
        elif line.startswith('Dataset features:'):
            dataset_features = line.split(':', 1)[1].strip()
    
    similar_datasets = Similarity().calc_sim(title, dataset_features)
    data1 = similar_datasets[0]
    data2 = similar_datasets[1]

    return data1, data2

st.set_page_config(layout="wide")

# Custom styles for the search bar and search button
search_bar_style = """
<style>
/* Style adjustments for the search input */
div.stTextInput > div > div > input {
    border-radius: 50px; /* Pill-shaped border */
    border: 1px solid #ced4da; /* Subtle border color */
    padding: 8px 20px; /* Adjusted padding */
}
/* Style adjustments for the search button */
div.stButton > button {
    border-radius: 20px; /* Slightly rounded corners for a pill shape */
    background-color: #FF763C;
    color: white;
    height: 40px; /* Adjust the height to match the search bar if necessary */
    border: none;
    outline: none; /* Remove the outline to maintain the appearance on focus */
    box-shadow: none; /* Remove default Streamlit shadow */
    padding: 0 15px; /* Horizontal padding for a wider pill shape */
    font-size: 16px; /* Optional font size adjustment for the button text */
    line-height: 40px; /* Adjust line height to match the button's height for vertical centering */
    cursor: pointer; /* Change the mouse cursor to indicate it's clickable */
    transform: translateY(70%); /* Center the button vertically */

}
</style>
"""

st.markdown(search_bar_style, unsafe_allow_html=True)

# Function to make the title nicer
def make_title_nice(title):
    title = title.replace('_', ' ').replace('-', ' ')
    title = title.title()
    return title

# Function to display dataset metadata and head of the dataset inside an expander
def display_metadata(dataset_name, metadata, similarity_score, file_path):
    with st.expander(f"{make_title_nice(dataset_name)} (Similarity Score: {similarity_score:.2f})"):
        st.markdown(f"**Source:** {metadata[0]}")
        st.markdown(f"**Views:** {metadata[1]}")
        st.markdown(f"**Year:** {metadata[2]}")
        st.markdown(f"**Description:** {metadata[4]}")
        st.markdown(f"**URL:** {metadata[5]}")
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df.head())  # Display the first few rows of the dataframe
        except Exception as e:
            st.error(f"An error occurred while loading the dataset: {e}")

# Page title
st.title('DataSphere.ai')
# Listing the first 5 CSV files in the specified directory
directory = "/Users/vinithamarupeddi/Desktop/catapult_hacks/catapulthacks/datasets"
# Creating two columns for the search bar and button
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("", placeholder="I want to create a machine learning model that predicts...", key="search")
with col2:
    button_placeholder = st.empty()

if button_placeholder.button('Discover', key="search_button"):
    st.session_state['user_input'] = user_input
    similar_datasets = generate_features(user_input)

    st.session_state['similar_datasets'] = similar_datasets

if 'similar_datasets' in st.session_state and st.session_state['similar_datasets'] and len(st.session_state['similar_datasets']) >= 1:
    data1 = st.session_state['similar_datasets'][0]
    
    dataset_name, similarity_score = data1
    dataset_path = os.path.join(directory, f"{dataset_name}.csv")
    metadata = json_data.get(dataset_name, ["Not Found"] * 6)

    display_metadata(dataset_name, metadata, similarity_score, dataset_path)


    st.text("Ask questions about the dataset:")
    col3, col4 = st.columns([3, 1])
    with col3:
        user_question = st.text_input("", placeholder="Enter your question here...", key="ai_question")
    with col4:
        ask_button = st.button('Ask', key='ask_button')

    if 'ai_question' in st.session_state and ask_button and st.session_state['ai_question']:
        agent = create_csv_agent(OpenAI(temperature=0), dataset_path, verbose=True)
        answer = agent.run(st.session_state['ai_question'])
        st.text_area("Answer", answer, height=150)
elif 'similar_datasets' in st.session_state and not st.session_state['similar_datasets']:
    st.error("Could not find similar datasets.")



def preprocess_column_names(col_name):
    col_name = col_name.strip().strip('"')
    
    col_name = col_name.replace("_", " ").rstrip('0123456789_')
    
    col_name = col_name.title()
    
    return col_name


# col1, col2 = st.columns([3, 1])

# with col1:
#     user_input = st.text_input("", "I want to create a machine learning model that predicts...", key="search")

# with col2:
#     # Using a placeholder to later display the button
#     # This helps to avoid issues related to Streamlit's execution order
#     button_placeholder = st.empty()

# if button_placeholder.button('Discover', key="search_button"):
#     # If button is pressed, find similar datasets
#     similar_datasets = generate_features(user_input)
    
#     # Make sure similar_datasets is a list or tuple of at least 1 element
#     if similar_datasets and len(similar_datasets) >= 1:
#         # Get the top result
#         data1 = similar_datasets[0]
        
#         # Display the top similar dataset
#         dataset_name, similarity_score = data1
#         dataset_path = os.path.join(directory, f"{dataset_name}.csv")
#         metadata = json_data.get(dataset_name, ["Not Found"] * 6)
#         display_metadata(dataset_name, metadata, similarity_score, dataset_path)
#     else:
#         st.error("Could not find similar datasets.")


# def preprocess_column_names(col_name):
#     # Remove any leading or trailing whitespace and quotes
#     col_name = col_name.strip().strip('"')

#     # Replace underscores and remove any trailing numbers with underscores
#     col_name = col_name.replace("_", " ").rstrip('0123456789_')

#     # Convert to title case
#     col_name = col_name.title()

#     return col_name

# Custom CSS to enlarge expander titles
st.markdown("""
<style>
[data-testid="stExpander"] .st-ae {
    font-size: 1.25rem;  /* Adjust the size as needed */
}
[data-testid="stExpander"] .st-bx {
    font-size: 1.25rem;  /* Adjust the size as needed */
}
</style>
""", unsafe_allow_html=True)
