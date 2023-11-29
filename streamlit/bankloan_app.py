# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# load the model from disk
import joblib

# model = joblib.load(r"./notebook/model.sav")
# model_threshold = joblib.load(r"./notebook/threshold.sav")

# Import python scripts
# from telco_input_preprocessing import preprocess


def main():
    # Setting Application title
    st.title("Bank Loan Amount Prediction App")


    option = st.selectbox(
        "How would you like to predict?", ("Single Predict", "Batch Predict"),
        index=None,
        placeholder="Select contact method...",
    )

    st.write(f'You selected: :green[{option}]')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.snow()

    # genre = st.radio(
    # "How would you like to predict?",
    # ["Single Predict", "Batch Predict"],
    # captions = ["Please fill up the form", "Please upload a csv file"])

    # if genre == 'Single Predict':
    #     st.write('Single')
    # else:
    #     st.write("Batch")

    if option == 'Single Predict':
        st.write('Single Predict')

    elif option == 'Batch Predict':
        st.subheader("Dataset upload")
        template = pd.read_csv('streamlit/template/customer_template.csv')
        get_template = st.download_button(
            label="Get template",
            file_name="customer_template.csv",
            data=template.to_csv(index=None),
            mime="text/csv",
        )
        uploaded_file = st.file_uploader(
            "Choose a file",
            help="Upload customer data. Save the file in csv format before uploading.",
        )
    
        if uploaded_file is not None:
            pass




if __name__ == "__main__":
    main()
