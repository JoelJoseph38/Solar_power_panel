import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained Random Forest model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions
def predict_defect(input_features):
    # Convert input into a numpy array
    input_array = np.array(input_features).reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_array)
    return prediction

# Streamlit app layout
def main():
    st.title('Solar Panel Defect Prediction')

    # User input options
    option = st.radio('Select Input Method', ['Upload File', 'Enter Features Manually'])

    if option == 'Upload File':
        uploaded_file = st.file_uploader("Upload file (CSV or Excel)", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.type == 'application/vnd.ms-excel' or uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)
            st.write('Uploaded File:')
            st.write(data)

            # Predict on uploaded file
            predictions = model.predict(data)
            st.write('Predictions:')
            st.write(predictions)

    elif option == 'Enter Features Manually':
        st.write('Enter the features to predict if a solar panel is defective or not.')

        # Input features
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
            # Make prediction
            input_features = [Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc]   
            prediction = predict_defect(input_features)
            # Display prediction
            st.write(f'The solar panel is {"defective" if prediction[0] == 1 else "non-defective"}.')

if __name__ == '__main__':
    main()
