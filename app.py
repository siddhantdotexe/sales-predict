import streamlit as st
import pandas as pd
import pickle

# --- Load trained model and data ---
with open('model.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('X_data.pkl', 'rb') as f:
    X = pickle.load(f)

# --- Streamlit UI ---
st.title("🛒 Big Mart Sales Prediction App")
st.write("Enter product details below to predict Item Outlet Sales.")

# Input form
with st.form("prediction_form"):
    Item_Identifier = st.text_input("Item Identifier", "FDG33")
    Item_Weight = st.number_input("Item Weight", min_value=0.0, value=15.5, step=0.1)
    Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    Item_Visibility = st.number_input("Item Visibility", min_value=0.0, value=0.05, step=0.01)
    Item_Type = st.text_input("Item Type", "Seafood")
    Item_MRP = st.number_input("Item MRP", min_value=0.0, value=225.5, step=0.5)
    Outlet_Identifier = st.text_input("Outlet Identifier", "OUT027")
    Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=2025, value=1985)
    Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    Outlet_Type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

    submit = st.form_submit_button("🔮 Predict Sales")

if submit:
    # Step 1: Create data dictionary
    custom_data = {
        'Item_Identifier': [Item_Identifier],
        'Item_Weight': [Item_Weight],
        'Item_Fat_Content': [Item_Fat_Content],
        'Item_Visibility': [Item_Visibility],
        'Item_Type': [Item_Type],
        'Item_MRP': [Item_MRP],
        'Outlet_Identifier': [Outlet_Identifier],
        'Outlet_Establishment_Year': [Outlet_Establishment_Year],
        'Outlet_Size': [Outlet_Size],
        'Outlet_Location_Type': [Outlet_Location_Type],
        'Outlet_Type': [Outlet_Type]
    }

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame(custom_data)

    # Step 3: Encode categorical values (consistent with training)
    input_df.replace({'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1}}, inplace=True)
    input_df.replace({'Outlet_Size': {'Small': 2, 'Medium': 1, 'High': 0}}, inplace=True)
    input_df.replace({'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}}, inplace=True)
    input_df.replace({'Outlet_Type': {
        'Grocery Store': 0,
        'Supermarket Type1': 1,
        'Supermarket Type2': 2,
        'Supermarket Type3': 3
    }}, inplace=True)

    # Step 4: Use representative encoded values from training data
    input_df['Item_Identifier'] = X['Item_Identifier'].mode()[0]
    input_df['Item_Type'] = 12
    input_df['Outlet_Identifier'] = X['Outlet_Identifier'].mode()[0]

    # Step 5: Predict
    prediction = regressor.predict(input_df)

    # Step 6: Display result
    st.success(f"💰 Predicted Sales: ${prediction[0]:.2f}")
    st.write("---")
    st.write("**Processed Input Data:**")
    st.dataframe(input_df)
