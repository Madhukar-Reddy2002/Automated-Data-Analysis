import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("Dataset Exploration App")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read data into DataFrame
        df = pd.read_csv(uploaded_file)

        # Display first 10 rows of the data
        st.subheader("First 10 rows of the dataset:")
        st.dataframe(df.head(10))

        # Display data description
        st.subheader("Data Description:")
        st.write(df.describe())

        # Handle missing values using interpolation
        df.interpolate(method='linear', inplace=True)

        # Apply label encoding to categorical columns
        label_encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':  # Check if the column is categorical
                df[col] = label_encoder.fit_transform(df[col])

        # Display encoded DataFrame
        st.subheader("Interpolated and Label Encoded Dataset:")
        st.dataframe(df)

        # Analyze column information again
        column_details = []
        for col in df.columns:
            column_details.append({
                "Column Name": col,
                "Missing Values": df[col].isnull().sum(),
                "Duplicate Values": sum(df[col].duplicated()),
                "Data Type": df[col].dtype
            })

        # Display column information in a table
        st.subheader("Updated Column Details:")
        st.dataframe(column_details)

if __name__ == "__main__":
    main()