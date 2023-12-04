import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from predict import predict_new
import base64

# load the model from disk
import joblib

# Import python scripts
# from telco_input_preprocessing import preprocess

# Set dataset path
dataset_path = "./dataset/train.csv"
bank_loan_df = pd.read_csv(dataset_path)

# Change column names for easier manipulation
bank_loan_df.columns = bank_loan_df.columns.str.replace(' ', '_') 

# List of employment types
emp_type = sorted([i for i in bank_loan_df['Type_of_Employment'].dropna().unique()])
emp_type.append("Others")

#profession Status
prof_type = sorted([i for i in bank_loan_df['Profession'].dropna().unique()])
prof_type.append("Others")

# Dictionary of property types
prop_type = {"Land": 1, "Residential": 2, "Commercial": 3, "Industrial": 4}


def set_background():
    background = """
    <style>
        body {
            background-image: url('streamlit/icon.jpg');
            background-size: cover;
        }
    </style>
    """
    st.markdown(background, unsafe_allow_html=True)



def single_predict():
    set_background()
    st.title("Bank Loan Amount Prediction App - Single Predict")
    st.write("Kindly fill in the applicant's details")

    customer_id = st.text_input('Customer ID', placeholder="Enter Customer ID...")
    name = st.text_input('Name', placeholder="Enter Customer's Name...")
    gender = st.selectbox("Gender", ("Male", "Female"), index=None, key="gender", help="Select Gender...")
    age = st.number_input('Age', min_value=18, max_value=100, value=None, placeholder="Input an Age...") 
    income = st.number_input('Monthly Income ($)', min_value=0.00, value=None, placeholder="Input Income...")
    income_stability = st.selectbox("Income Stability", ("Low", "High"), index=None, key="income_stability", help="Select Income Stability...")
    profession = st.selectbox('Profession Status', prof_type, index=None, placeholder="Input Profession Status...")
    type_of_employment = st.selectbox('Type of Employment', emp_type, index=None, key="type_of_employment", help="Input Employment Type...")
    location = st.selectbox("Work Location", ("Urban", "Semi-Urban", "Rural"), index=None, key="location", help="Select Work location...")
    loan_amount_request = st.number_input('Loan Amount Requested ($)', min_value=0.00, value=None, placeholder="Input Loan Amount Requested...")  
    current_loan_expenses = st.number_input('Current Loan Expense ($)', min_value=0.00, value=None, placeholder="Input Current Loan Expense...")
    expense_t1 = st.selectbox("Expense Type 1 (include Other Loan Payments)", ("Yes", "None"), index=None, key="expense_t1", help="Applicant Has Type 1 Expenses?") 
    expense_t2 = st.selectbox("Expense Type 2 (include Insurance)", ("Yes", "None"), index=None, key="expense_t2", help="Applicant Has Type 2 Expenses?") 
    dependents = st.number_input('Number of dependents', min_value=0, value=None, placeholder="Input Number of dependents...")
    credit_score = st.number_input('Credit Score', min_value=0, value=None, placeholder="Input Credit Score...")
    num_of_defaults = st.selectbox("History of defaults", ("Yes", "None"), index=None, key="num_of_defaults", help="History of defaults") 
    has_credit_card = st.selectbox("Credit Card Status", ("Unpossessed", "Active", "Inactive"), index=None, key="has_credit_card", help="Select Credit Card Status...")
    # property_id = st.number_input('Property Id', min_value=0, max_value=999,value=None, placeholder="Input Property Id...")  
    property_age = st.number_input('Property Age (in years)', min_value=0, value=None, placeholder="Input Property Age (in years)")
    property_type = st.selectbox("Property Type", prop_type, index=None, key="property_type", help="Select Property Type")  
    property_location = st.selectbox("Property Location", ("Urban", "Semi-Urban", "Rural"), index=None, key="property_location", help="Select property location...")
    co_applicant = st.selectbox("Presence of Co-applicant", ("Yes", "None"), index=None, key="co_applicant",help="Presence of Co-applicant...") 
    property_price = st.number_input('Property Price ($)', min_value=0.00, value=None, placeholder="Input Property Price...")

    st.markdown("<h3></h3>", unsafe_allow_html=True)


    features_df = pd.DataFrame.from_dict({
                'Customer_ID': [customer_id], 
                'Name': [name], 
                'Gender': [gender], 
                'Age': [age], 
                'Income_(USD)': [income],
                'Income_Stability': [income_stability], 
                'Profession': [profession], 
                'Type_of_Employment': [type_of_employment], 
                'Location': [location],
                'Loan_Amount_Request_(USD)': [loan_amount_request], 
                'Current_Loan_Expenses_(USD)': [current_loan_expenses],
                'Expense_Type_1': [expense_t1],
                'Expense_Type_2': [expense_t2], 
                'Dependents': [dependents], 
                'Credit_Score': [credit_score],
                'History_of_Defaults': [num_of_defaults], 
                'Has_Active_Credit_Card': [has_credit_card],
                'Property_Age': [property_age], 
                'Property_Type': [prop_type[property_type] if property_type in prop_type else None], 
                'Property_Location': [property_location], 
                'Co-Applicant': [co_applicant],
                'Property_Price': [property_price]
            })

    st.markdown("<h3>Overview of input:</h3>", unsafe_allow_html=True)
    st.dataframe(features_df)



    if st.button('Predict'):
        # Validate and process the inputs for prediction
        missing_fields = []

        if not customer_id:
            missing_fields.append("Customer ID")
        if not name:
            missing_fields.append("Name")
        if not gender:
            missing_fields.append("Gender")
        if not age:
            missing_fields.append("Age")
        if income == None:
            missing_fields.append("Monthly Income ($)")
        if income_stability== None:
            missing_fields.append("Income Stability")
        if profession == None:
            missing_fields.append("Profession")
        if type_of_employment== None:
            missing_fields.append("Type of Employment")
        if location== None:
            missing_fields.append("Work Location")
        if loan_amount_request== None:
            missing_fields.append("Loan Amount Requested ($)")
        if current_loan_expenses== None:
            missing_fields.append("Current Loan Expense ($)")
        if expense_t1== None: 
            missing_fields.append("Expense Type 1")
        if expense_t2== None: 
            missing_fields.append("Expense Type 2")
        if dependents== None:
            missing_fields.append("Number of Dependents")
        if credit_score== None:
            missing_fields.append("Credit Score")
        if num_of_defaults== None:
            missing_fields.append("History of Defaults")
        if has_credit_card== None:
            missing_fields.append("Credit Card Status")
        if property_age== None:
            missing_fields.append("Property Age")
        if property_type== None:
            missing_fields.append("Property Type")
        if property_location== None:
            missing_fields.append("Property Location")
        if co_applicant== None:
            missing_fields.append("Presence of Co-applicants")
        if property_price == None:
            missing_fields.append("Property Price")
            
        if missing_fields:
            st.error(f"Please fill in the following fields: {', '.join(missing_fields)}")         


        features_df = pd.DataFrame.from_dict({
                'Customer_ID': [customer_id], 
                'Name': [name], 
                'Gender': [gender], 
                'Age': [age], 
                'Income_(USD)': [income],
                'Income_Stability': [income_stability], 
                'Profession': [profession], 
                'Type_of_Employment': [type_of_employment], 
                'Location': [location],
                'Loan_Amount_Request_(USD)': [loan_amount_request], 
                'Current_Loan_Expenses_(USD)': [current_loan_expenses],
                'Expense_Type_1': [expense_t1],
                'Expense_Type_2': [expense_t2], 
                'Dependents': [dependents], 
                'Credit_Score': [credit_score],
                'History_of_Defaults': [num_of_defaults], 
                'Has_Active_Credit_Card': [has_credit_card],
                'Property_Age': [property_age], 
                'Property_Type': [prop_type[property_type] if property_type in prop_type else None], 
                'Property_Location': [property_location], 
                'Co-Applicant': [co_applicant],
                'Property_Price': [property_price]
            })



        st.markdown("<h3></h3>", unsafe_allow_html=True)

        #Display only the prediction amount if there is no missing fields
        if len(missing_fields) == 0:
            data = {
                'Customer ID': [customer_id],
                'Name': [name],
                'Gender': [gender[0]],
                'Age': [age],
                'Income (USD)': [income],
                'Income Stability': [income_stability],
                'Profession': [profession],
                'Type of Employment': [type_of_employment],
                'Location': [location],
                'Loan Amount Request (USD)': [loan_amount_request],
                'Current Loan Expenses (USD)': [current_loan_expenses],
                'Expense Type 1': [expense_t1[0]],
                'Expense Type 2': [expense_t2[0]],
                'Dependents': [dependents],
                'Credit Score': [credit_score],
                'No. of Defaults': [0 if num_of_defaults == "None" else 1],
                'Has Active Credit Card': [has_credit_card],
                'Property Age': [property_age],
                'Property Type': [prop_type[property_type] if property_type in prop_type else None],
                'Property Location': [property_location],
                'Co-Applicant': [0 if co_applicant == "None" else 1],
                'Property Price': [property_price]
                }
            prediction = predict_new(data)[0][0]
            # st.write("The customer can loan a maximum amount of: :green[$] ", prediction)
            # Assuming 'prediction' contains the predicted loan amount
            # st.markdown(f'<p style="font-size:24px">The customer can loan a maximum amount of: <p style="font-size:24px; color:green;"> $ {prediction:.2f}</p>', unsafe_allow_html=True)
            # Assuming 'prediction' contains the predicted loan amount
            st.markdown(f'<div style="border: 2px solid green; padding: 10px; border-radius: 10px; text-align: center;"> \
                    <p style="font-size:20px">The customer can loan a maximum amount of:</p> \
                    <p style="font-size:30px; color:green; font-weight: bold;"> $ {prediction:.2f}</p> \
                </div>', unsafe_allow_html=True)




def batch_predict():
    st.title("Bank Loan Amount Prediction App - Batch Predict")
    st.subheader("Dataset upload")

    # Download the csv file containing the prediction
    template = pd.read_csv('streamlit/template/customer_template.csv')

    get_template = st.download_button(
        label="Get template",
        file_name="customer_loan_application_template.csv",
        data=template.to_csv(index=None),
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        help="Upload customer data. Save the file in CSV format before uploading.",
    )

    #a file is uploaded
    if uploaded_file is not None:
        
        #add code that validates if file submitted is csv
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.markdown(f'<div style="border: 2px solid green; padding: 10px; border-radius: 10px; text-align: center;"> \
                <p style="font-size:20px">Check the initial column for the maximum loan amount.</p> \
            </div>', unsafe_allow_html=True)
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        data = pd.read_csv(uploaded_file)
        predictions = predict_new(data.to_dict('list'))

        predictions_df = pd.DataFrame(predictions, columns=["Maximum Loan Amount ($)"])
        prediction_and_x_values_df = pd.concat([predictions_df, data], axis=1)

        prediction_and_x_values_df["Maximum Loan Amount ($)"] = prediction_and_x_values_df["Maximum Loan Amount ($)"].round(2).apply(lambda x: '$ {:,.2f}'.format(x))

        print(prediction_and_x_values_df.round(1))
        st.write(prediction_and_x_values_df)

        

def main():
    # Setting Application title
    st.set_page_config(page_title="Bank Loan Amount Prediction App", layout="wide")

    
    # image = Image.open('Loan_Photo.jpg')
    # st.image(image)
    video_file = open("Loan_video.mp4", "rb").read()
    video_encoded = base64.b64encode(video_file).decode('utf-8')
    video_tag = f'<video width="100%" controls autoplay loop muted src="data:video/mp4;base64,{video_encoded}" type="video/mp4"></video>'

    st.markdown(video_tag, unsafe_allow_html=True)

    # Hide the Streamlit menu
    hide_menu_style = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    # Add a sidebar
    st.sidebar.title("Choose Prediction Mode")
    option = st.sidebar.selectbox(
        "Select Prediction Type", ("Single Predict", "Batch Predict"), index=None, placeholder="Select Prediction Type..."
    )

    st.sidebar.write(f'You selected: :green[{option}]')
    st.sidebar.markdown("<h3></h3>", unsafe_allow_html=True)

    # Display content based on the selected option
    if option == 'Single Predict':
        single_predict()
    elif option == 'Batch Predict':
        batch_predict()


if __name__ == "__main__":
    main()
