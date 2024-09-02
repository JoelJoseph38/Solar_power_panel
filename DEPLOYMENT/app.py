import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained Random Forest model from the pickle file
@st.cache(allow_output_mutation=True)
def load_model():
    with open('Random_Forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to preprocess and make predictions
def predict_defect(input_data):
    # Columns expected by the model
    expected_columns = ['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', 'If', 'Vabc']

    # Remove any columns not in the expected columns list
    input_data = input_data[expected_columns]

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title('Solar Panel Defect Prediction')

    # User input options
    option = st.radio('Select Input Method:', ['Upload File', 'Enter Features Manually'])

    if option == 'Upload File':
        uploaded_file = st.file_uploader("Upload file (CSV or Excel)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.type.startswith('text/csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                st.write('Uploaded File:')
                st.write(data)

                # Predict on uploaded file after preprocessing
                predictions = predict_defect(data)
                # Append predictions to the DataFrame
                data['Predictions'] = predictions
                data['Predictions'] = data['Predictions'].apply(lambda x: 'defective' if x == 1 else 'non-defective')
                st.write('Data with Predictions:')
                st.write(data)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

    elif option == 'Enter Features Manually':
        st.write('Enter the features to predict if a solar panel is defective or not.')

        # Input features according to the trained model's requirements
        Ipv = st.number_input('Ipv')
        Vpv = st.number_input('Vpv')
        Vdc = st.number_input('Vdc')
        ia = st.number_input('ia')
        ib = st.number_input('ib')
        ic = st.number_input('ic')
        va = st.number_input('va')
        vb = st.number_input('vb')
        vc = st.number_input('vc')
        Iabc = st.number_input('Iabc')
        If = st.number_input('If')
        Vabc = st.number_input('Vabc')

        # Predict button
        if st.button('Predict'):
            input_features = pd.DataFrame([[
                Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc
            ]], columns=['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', 'If', 'Vabc'])

            prediction = predict_defect(input_features)
            result = 'defective' if prediction[0] == 1 else 'non-defective'
            st.write(f'The solar panel is {result}.')

if __name__ == '__main__':
    main()
