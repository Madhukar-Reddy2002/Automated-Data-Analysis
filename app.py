import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def remove_outliers(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def main():
    st.title("👻 Dataset Exploration and Modeling App 👻")

    # Set custom styles
    styles = """
    <style>
    .stFileUploader {
        background-color: #ffe5ec; /* light pink */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #b30059; /* dark pink */
    }
    .stSelectbox {
        background-color: #e6f9ff; /* light blue */
        padding: 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        color: #0077b6; /* dark blue */
    }
    .downloadLink {
        background-color: #d4edda; /* light green */
        padding: 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        color: #155724; /* dark green */
        text-decoration: none;
    }
    </style>
    """
    st.markdown(styles, unsafe_allow_html=True)

    # Upload dataset
    st.write("*Rubbing my hands together excitedly* 🧞‍♂️ Hey there! I'm your friendly AI genie, here to make your modeling dreams come true with a sprinkle of magic! 🪄")
    st.write("First things first, let's get that dataset uploaded! Just click the handy button below, and we'll get the ball rolling. Once your file is loaded, I'll give you a sneak peek at the first few rows, so you can get a feel for your data. 👀")

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

        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Impute numeric columns with mean and categorical columns with mode
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

        # Remove outliers
        st.write("Removing outliers...")
        df = remove_outliers(df, numeric_cols)

        # Apply label encoding to categorical columns
        label_encoder = OrdinalEncoder()
        encoded_df = df.copy()
        encoded_df[categorical_cols] = label_encoder.fit_transform(df[categorical_cols])

        # Display encoded DataFrame
        st.subheader("Processed Dataset:")
        st.dataframe(encoded_df)

        # Analyze column information again
        column_details = []
        for col in encoded_df.columns:
            column_details.append({
                "Column Name": col,
                "Missing Values": encoded_df[col].isnull().sum(),
                "Duplicate Values": sum(encoded_df[col].duplicated()),
                "Data Type": encoded_df[col].dtype
            })

        # Display column information in a table
        st.subheader("Column Details:")
        st.dataframe(column_details)

        # Show correlation matrix as heatmap
        st.subheader("Correlation Matrix Heatmap")

        # Create figure and axes explicitly
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(encoded_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Ask user to choose target column
        st.write("But the real fun begins when you select your target variable! Use the dropdown menu below to choose your target, and I'll show you which columns have a high correlation with it, so you can focus on the most important features.")

        target_variable = st.selectbox("Select Target Variable:", encoded_df.columns)

        # If the user has selected a target variable, proceed
        if target_variable:
            st.subheader(f"Correlation with '{target_variable}'")
            correlation_with_target = encoded_df.corr()[target_variable]
            high_correlation_cols = correlation_with_target[(correlation_with_target > 0.6) | (correlation_with_target < -0.6)].index.tolist()

            # Remove the target variable itself from the list
            high_correlation_cols.remove(target_variable)

            if high_correlation_cols:
                st.write("Columns with high correlation (> 0.6 or < -0.6) with the selected target variable:")
                st.write(high_correlation_cols)

                # Prepare data for modeling
                X = encoded_df[high_correlation_cols]
                y = encoded_df[target_variable]

                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create and evaluate different models
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor()
                }

                model_performance = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    model_performance[name] = {"RMSE": rmse, "R-squared": r2}

                # Select the best performing model
                best_model_name = min(model_performance, key=lambda x: model_performance[x]["RMSE"])
                best_model = models[best_model_name]
                best_model_rmse = model_performance[best_model_name]["RMSE"]
                best_model_r2 = model_performance[best_model_name]["R-squared"]

                st.write(f"Best performing model: {best_model_name}")
                st.write(f"Root Mean Squared Error (RMSE): {best_model_rmse}")
                st.write(f"R-squared: {best_model_r2}")

                # Display model statistics
                st.subheader("Model Statistics:")
                if isinstance(best_model, LinearRegression):
                    st.write("Intercept:", best_model.intercept_)
                    st.write("Coefficients:")
                    for i, coef in enumerate(best_model.coef_):
                        st.write(f"{high_correlation_cols[i]}: {coef}")
                else:
                    st.write("The best model does not have an intercept or coefficients.")

                # Save the best model
                model_filename = "best_model.joblib"
                joblib.dump(best_model, model_filename)

                # Provide download link for the model
                st.write("But wait, there's more! I'll save your winning model to a file, and even provide a handy download link, so you can take it with you wherever you go. 📥")
                st.markdown(f'[<span class="downloadLink">Download Trained Model</span>]({model_filename})', unsafe_allow_html=True)

                # Visualize model performance
                st.subheader("Model Performance Visualization")
                for col in high_correlation_cols:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=X_test[col], y=y_test, color='blue', label='Actual')
                    sns.lineplot(x=X_test[col], y=best_model.predict(X_test), color='red', label='Predicted')
                    plt.title(f"{col} vs. {target_variable}")
                    plt.xlabel(col)
                    plt.ylabel(target_variable)
                    st.pyplot(plt)

            else:
                st.write("No columns have high correlation (> 0.6 or < -0.6) with the selected target variable.")

if __name__ == "__main__":
    main()