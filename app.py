import streamlit as st
import pickle
import pandas as pd
import sklearn
model_data = pickle.load(open('thyroid_model.pkl', 'rb'))
model = model_data['model']
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = model_data['label_encoders']

numerical_feature=['Age','Thyroid Function','Physical Examination','Adenopathy','Pathology','Risk','T','N','M','Stage','Response']
categorical_columns=['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function','Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk','T', 'N', 'M', 'Stage', 'Response']

st.title("Thyroid Detection")

def get_user_input():
    # Collect numerical inputs
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    #thyroid_function = st.number_input('Thyroid Function', min_value=0.0, max_value=100.0, step=0.1)
    #physical_examination = st.number_input('Physical Examination', min_value=0.0, max_value=100.0, step=0.1)
    #adenopathy = st.number_input('Adenopathy', min_value=0.0, max_value=100.0, step=0.1)
    #pathology = st.number_input('Pathology', min_value=0.0, max_value=100.0, step=0.1)
    #risk = st.number_input('Risk', min_value=0.0, max_value=100.0, step=0.1)
    #t = st.number_input('T', min_value=0.0, max_value=10.0, step=0.1)
    #n = st.number_input('N', min_value=0.0, max_value=10.0, step=0.1)
    #m = st.number_input('M', min_value=0.0, max_value=10.0, step=0.1)
    #stage = st.number_input('Stage', min_value=0.0, max_value=10.0, step=0.1)
    #response = st.number_input('Response', min_value=0.0, max_value=100.0, step=0.1)

    # Collect categorical inputs
    gender = st.selectbox('Gender', ['M', 'F'])
    smoking = st.selectbox('Smoking', ['Yes', 'No'])
    hx_smoking = st.selectbox('Hx Smoking', ['Yes', 'No'])
    hx_radiotherapy = st.selectbox('Hx Radiothreapy', ['Yes', 'No'])
    thyroid_function=st.selectbox('Thyroid Fuction',['Euthyroid','Clinical Hyperthyroidism','Clinical Hypothyroidism','Subclinical Hyperthyroidism','Subclinical Hypothyroidism'])
    physical_examination=st.selectbox('Physical Examination',['Single nodular goiter-left','Multinodular goiter','Single nodular goiter-right','Normal','Diffuse goiter'])
    adenopathy=st.selectbox('Adenopathy',['No','Right','Extensive','Left','Bilateral','Posterior'])
    pathology=st.selectbox('Pathology',['Micropapillary','Papillary','Follicular','Hurthel cell'])
    focality = st.selectbox('Focality', ['Uni-Focal', 'Multi-Focal'])
    risk=st.selectbox('Risk',['Low','Intermediate','High'])
    t=st.selectbox('T',['T1a','T1b','T2','T3a','T3b','T4a','T4b'])
    n=st.selectbox('N',['N0','N1b','N1a'])
    m=st.selectbox('M',['M0','M1'])
    stage=st.selectbox('Stage',['I','II','IVB','III','IVA'])
    response=st.selectbox('Response',['Indeterminate','Excellent','Structural Incomplete','Biochemical Incomplete'])
    # Store the user inputs in a dictionary
    user_data = {
        'Age': [age],
        'Gender': [gender],
        'Smoking': [smoking],
        'Hx Smoking': [hx_smoking],
        'Hx Radiothreapy': [hx_radiotherapy],
        'Thyroid Function': [thyroid_function],
        'Physical Examination': [physical_examination],
        'Adenopathy': [adenopathy],
        'Pathology': [pathology],
        'Focality': [focality],
        'Risk': [risk],
        'T': [t],
        'N': [n],
        'M': [m],
        'Stage': [stage],
        'Response': [response]
    }
    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(user_data)
    return features

# Get user input
user_input = get_user_input()

for col in categorical_columns:
    user_input[col] = label_encoders[col].transform(user_input[col])

user_input[numerical_feature] = scaler.transform(user_input[numerical_feature])

if st.button("Predict"):
    prediction = model.predict(user_input)
    result = "Recurred" if prediction[0] == 1 else "Not Recurred"
    st.write(f"Prediction: {result}")

