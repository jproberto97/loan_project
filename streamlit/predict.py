import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn

def predict_new(values_dict):
    """
    Input: dictionary
    
    Example dictionary for 3 rows:

    {'Customer ID': ['C-17688', 'C-23855', 'C-26934'],
    'Name': ['Polly Crumpler', 'Nathalie Olivier', 'Kenny Ankrom'],
    'Gender': ['F', 'M', 'F'],
    'Age': [60, 43, 38],
    'Income (USD)': [1234.92, 2361.56, 1296.07],
    'Income Stability': ['Low', 'Low', 'Low'],
    'Profession': ['State servant', 'Working', 'Working'],
    'Type of Employment': ['Secretaries', 'Laborers', 'Cooking staff'],
    'Location': ['Rural', 'Semi-Urban', 'Rural'],
    'Loan Amount Request (USD)': [34434.72, 152561.34, 35141.99],
    'Current Loan Expenses (USD)': [181.48, 697.67, 155.95],
    'Expense Type 1': ['N', 'Y', 'N'],
    'Expense Type 2': ['N', 'Y', 'Y'],
    'Dependents': [2.0, 2.0, 3.0],
    'Credit Score': [684.12, 637.29, 705.29],
    'No. of Defaults': [1, 0, 1],
    'Has Active Credit Card': ['Inactive', 'Unpossessed', 'Active'],
    'Property Age': [1234.92, 2361.56, 1296.07],
    'Property Type': [2, 1, 4],
    'Property Location': ['Rural', 'Semi-Urban', 'Rural'],
    'Co-Applicant': [1, 1, 1],
    'Property Price': [43146.82, 221050.8, 54903.44],
    'Loan Sanction Amount (USD)': [22382.57, 0.0, 22842.29]}

    Output: numpy array

    Example output for 3 rows:

    array([[26180.29],
       [99073.36],
       [25242.82]], dtype=float32)
    """

    column_label_dictionary = {'Monthly Income ($)':'Income (USD)', 'Work Location':'Location', 'Loan Amount Requested($)':'Loan Amount Request (USD)', 'Current Loan Expenses($)':'Current Loan Expenses (USD)', 'Expense Type 1(include Loan Payments and Insurance)':'Expense Type 1', 'Expense Type 2(include Travel and Subscription Services)':'Expense Type 2', 'Number of dependents':'Dependents', ' History of defaults':'No. of Defaults', 'Credit Card Status':'Has Active Credit Card', 'Property Age(in Years)':'Property Age', 'Number of Co-applicants':'Co-Applicant', 'Property Price($)':'Property Price'}
    
    bank_loan_df = pd.DataFrame.from_dict(values_dict)
    bank_loan_df.rename(columns=column_label_dictionary, inplace=True)

    bank_loan_df.columns = bank_loan_df.columns.str.replace(' ', '_') 
    unnecessary_columns = ['Customer_ID', 'Name']
    bank_loan_df.drop(unnecessary_columns, axis=1, inplace=True)
    bank_loan_df['Property_Age'] = bank_loan_df['Property_Age'] * 365.25

    ohe_df = pd.get_dummies(bank_loan_df, dtype=int)

    with open('./prediction_files/columns_list.pkl','rb') as f:
        columns = pickle.load(f)

    columns_to_drop = [column for column in ohe_df.columns if column not in columns]
    ohe_df = ohe_df.astype(float)
    ohe_df.drop(columns_to_drop, axis=1, inplace=True)

    columns_to_add = [column for column in columns if column not in ohe_df.columns]
    for column in columns_to_add:
        ohe_df[column] = 0

    positive_skew = ['Income_(USD)', 'Loan_Amount_Request_(USD)', 'Current_Loan_Expenses_(USD)', 'No._of_Defaults']
    negative_skew = ['Co-Applicant']
    for column in positive_skew:
        ohe_df[column] = ohe_df[column].apply(np.log1p)
    for column in negative_skew:
        ohe_df[column] = ohe_df[column].apply(lambda x : x**3)

    with open('./prediction_files/preprocessing_scaler.pkl','rb') as f:
        scaler = pickle.load(f)

    ohe_df = ohe_df[columns]
    ohe_df = pd.DataFrame(scaler.transform(ohe_df), columns= ohe_df.columns, index=ohe_df.index)
    x_test = torch.Tensor(ohe_df.values)

    activation = nn.ReLU()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim

            self.activation = activation

            self.multiple_layers = nn.Sequential(
                nn.Linear(self.in_dim, 10),
                self.activation,

                nn.Linear(10, 7),
                self.activation,

                nn.Linear(7, 3),
                self.activation,

                nn.Linear(3, 2),
                self.activation,

                nn.Linear(2, self.out_dim),
                self.activation,
            )
            
        
        def forward(self, x):
            
            y = self.multiple_layers(x)
        
            return y
    
    in_dim = x_test.shape[1]
    out_dim = 1

    model = NeuralNetwork(in_dim, out_dim)
    state = torch.load("./models/dlmodel_0.98433_neil")
    model.load_state_dict(state['state_dict'])
    model.to(device)

    with open('./prediction_files/relu_scaler.pkl','rb') as f:
        scaler = pickle.load(f)

    predictions = scaler.inverse_transform(model.forward(x_test).detach().numpy())
    
    np.set_printoptions(precision=2)
    return predictions