# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from predict import predict_new

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

        #placeholders; please edit
        customer_id = 'C-17688'

        name = 'Polly Crumpler'

        gender = 'F'

        age = 60
        
        income = 1234.92
        
        income_stability = 'Low'
        
        profession = 'State servant'
        
        type_of_employment = 'Secretaries'
        
        location = 'Rural'
        
        loan_amt_request = 34434.72
        
        current_loan_expenses = 181.48
        
        expense_type_1 = 'N'
        
        expense_type_2 = 'N'
        
        dependents = 2
        
        credit_score = 684.12
        
        defaults = 1
        
        credit_card = 'Inactive'
        
        property_id = 491
        
        property_age = 1234.92
        
        property_type = 2
        
        property_location = 'Rural'
        
        co_applicant = 1
        
        property_price = 43146.82
            
        data = {
            'Customer ID': [customer_id],
            'Name': [name],
            'Gender': [gender],
            'Age': [age],
            'Income (USD)': [income],
            'Income Stability': [income_stability],
            'Profession': [profession],
            'Type of Employment': [type_of_employment],
            'Location': [location],
            'Loan Amount Request (USD)': [loan_amt_request],
            'Current Loan Expenses (USD)': [current_loan_expenses],
            'Expense Type 1': [expense_type_1],
            'Expense Type 2': [expense_type_2],
            'Dependents': [dependents],
            'Credit Score': [credit_score],
            'No. of Defaults': [defaults],
            'Has Active Credit Card': [credit_card],
            'Property ID': [property_id],
            'Property Age': [property_age],
            'Property Type': [property_type],
            'Property Location': [property_location],
            'Co-Applicant': [co_applicant],
            'Property Price': [property_price]
            }
        
        prediction = predict_new(data)[0][0]


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
            data = pd.read_csv(uploaded_file)
            predictions = predict_new(data.to_dict('list'))
            predictions_df = pd.DataFrame(predictions, columns=["Loan Sanction Amount (USD)"])
            prediction_and_x_values_df = pd.concat([predictions_df, data], axis=1)




if __name__ == "__main__":
    main()
