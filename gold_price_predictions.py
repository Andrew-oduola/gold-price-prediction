import numpy as np
from PIL import Image
import streamlit as st
import pickle


# Load the model
def load_model():
    with open('gold_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Predict heart disease
def predict_gold_price(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    return prediction

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Gold Price Prediction", page_icon="", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("Gold Price Prediction")
    st.markdown("This app predicts the price of gold based on the input features.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
        st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")

        st.title("About")
        st.info("This app uses a machine learning model to predict the price of gold based on the input features.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    
    st.title("Market Index Inputs")

    st.markdown("Enter the latest values for key market indices and commodities.")

    spx = st.number_input(
        "📈 S&P 500 Index", 
        min_value=0.0, 
        max_value=5000.0, 
        value=1000.0, 
        step=100.0,
        help="Enter the current S&P 500 value (0 - 5000)."
    )

    usd_euro = st.number_input(
        "💱 USD to EURO Exchange Rate", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=1.0,
        help="Enter the USD/EURO index value (0 - 100)."
    )

    uso = st.number_input(
        "🛢️ Crude Oil (USO)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=1.0,
        help="Enter the current price of crude oil (USO ETF)."
    )

    slv = st.number_input(
        "🥈 Silver (SLV)", 
        min_value=0.0, 
        max_value=100.0, 
        value=50.0, 
        step=1.0,
        help="Enter the current price of silver (SLV ETF)."
    )

    input_data = [spx, uso, slv, usd_euro]
    # input_data = np.array(input_data).reshape(1, -1)


    # Prediction button
    if st.button("Predict"):
        prediction = predict_gold_price(input_data)
        result_placeholder.success("Predicted Gold Price: $%.2f" % prediction[0])
        st.success("Predicted Gold Price: $%.2f" % prediction[0])
        
        
if __name__ == "__main__":
    main()