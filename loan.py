import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter


# Streamlit Configurations
st.set_page_config(layout="wide")
st.title("Loan Approval Analysis and Prediction App")

# Load Dataset
df = pd.read_csv("Loan approval prediction.csv")
df = df.drop(columns=["id"])  # Drop unnecessary columns
df['Age Level'] = pd.cut(
    df['person_age'],
    bins=[19, 25, 35, 46],
    labels=['20-24', '25-34', '35-45'],
    right=True
)
df['loan_status'] = df['loan_status'].replace({0: 'Not Approved', 1: 'Approved'})
df['loan_grade'] = df['loan_grade'].replace({
    'A': 'Excellent Credit',
    'B': 'Good Credit',
    'C': 'Fair Credit',
    'D': 'Poor Credit',
    'E': 'Very Poor Credit',
    'F': 'Creditworthiness',
    'G': 'No Credit Score'  # or any other meaning for 'G'
})
# Sidebar Options
option = st.sidebar.selectbox("Choose a Section", ["Home", "EDA", "ML"])

if option == "Home":
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write("### Summary of Numerical Columns")
    st.write(df.describe())
    st.write("### Summary of Categorical Columns")
    st.write(df.describe(include="object"))

elif option == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    
    # Add Outlier Analysis Section
    with st.expander("Advanced Options: Outlier Analysis", expanded=False):
        st.subheader("Outlier Visualizations")
        numerical_columns = [
            "person_age", "person_income", "person_emp_length", 
            "loan_amnt", "loan_int_rate", "loan_percent_income", 
            "cb_person_cred_hist_length"
        ]
        selected_column = st.selectbox("Select Column for Outlier Analysis", numerical_columns)
        
        df_age_above_45 = df[df["person_age"] <= 45]
        filtered_age_income = df_age_above_45[df_age_above_45["person_income"] <= 150000]
        filtered_df = filtered_age_income[filtered_age_income["person_emp_length"] <= 15]
        df_after_outliers = filtered_df[filtered_df["loan_amnt"] <= 25000]

        # Create tabs for visualization
        tab1, tab2, tab3 = st.tabs(["Box Plot", "Histogram Before Outliers", "Histogram After Outliers"])

        # Tab 1: Box Plot
        with tab1:
            st.write(f"### Box Plot for {selected_column}")
            fig_box, ax_box = plt.subplots()
            sns.boxplot(x=df[selected_column], ax=ax_box, color="skyblue")
            ax_box.set_title(f"Box Plot of {selected_column}")
            st.pyplot(fig_box)

        # Tab 2: Histogram Before Handling Outliers
        with tab2:
            st.write(f"### Histogram for {selected_column} (Before Handling Outliers)")
            fig_hist_before, ax_hist_before = plt.subplots()
            sns.histplot(df[selected_column], kde=True, bins=30, color="blue", ax=ax_hist_before)
            ax_hist_before.set_title(f"Distribution of {selected_column} (Before Handling Outliers)")
            ax_hist_before.set_xlabel(selected_column)
            ax_hist_before.set_ylabel("Frequency")
            st.pyplot(fig_hist_before)

        # Tab 3: Histogram After Handling Outliers
        with tab3:
            st.write(f"### Histogram for {selected_column} (After Handling Outliers)")
            fig_hist_after, ax_hist_after = plt.subplots()
            sns.histplot(df_after_outliers[selected_column], kde=True, bins=30, color="green", ax=ax_hist_after)
            ax_hist_after.set_title(f"Distribution of {selected_column} (After Handling Outliers)")
            ax_hist_after.set_xlabel(selected_column)
            ax_hist_after.set_ylabel("Frequency")
            st.pyplot(fig_hist_after)
    
    
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Function to format column names: Capitalize the first letter of each word and replace underscores with spaces
    def format_column_name(col):
        return col.replace('_', ' ').title()

    # Apply formatting to the columns in the select box
    formatted_columns = [format_column_name(col) for col in categorical_columns]

    # Create a select box for the user to choose a categorical column
    selected_column = st.selectbox("Select a categorical column", formatted_columns)

    # Display the value counts for the selected column and plot a bar chart
    if selected_column:
        # Convert the formatted name back to the original column name
        original_column_name = categorical_columns[formatted_columns.index(selected_column)]
        
        # Format the column name for the subheader
        formatted_column_name = selected_column
        
        # Display the subheader with the formatted column name
        st.subheader(f"Value Counts of {formatted_column_name}")
        
        st.write(f"Below is the value count distribution for {formatted_column_name}:")
        
        value_counts = df[original_column_name].value_counts()

        # Plotting the bar chart using seaborn
        fig, ax = plt.subplots()
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette="Set2")
        ax.set_title(f"Value Counts of {formatted_column_name}")
        
        # Formatting the axis labels: Capitalize the first letter and replace underscores with spaces
        ax.set_xlabel(formatted_column_name)
        ax.set_ylabel("Count")

        # Adjust x-axis labels to avoid overlap
        plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and resize labels
        st.pyplot(fig)
    
    
    # Loan Status Count by Age Level
    st.subheader("Loan Status Count by Age Level")
    grouped_data = df.groupby(["Age Level", "loan_status"])["loan_status"].count().reset_index(name="count")

    fig, ax = plt.subplots()
    sns.barplot(data=grouped_data, x="Age Level", y="count", hue="loan_status", palette="Set2", ax=ax)
    ax.set_title("Loan Status Count by Age Level")
    ax.set_xlabel("Age Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Loan Intent Distribution Across Age Levels
    st.subheader("Loan Intent Distribution Across Age Levels")
    ages_level = df.groupby(["Age Level", "loan_intent"])["loan_intent"] \
        .count() \
        .reset_index(name="count")
    fig = px.bar(
        ages_level,
        x="Age Level",
        y="count",
        color="loan_intent",
        text="count",
        title="Loan Intent Distribution Across Age Levels",
        labels={"count": "Loan Intent Count", "loan_intent": "Loan Intent"},
        barmode="group",
        hover_name="loan_intent",
        color_discrete_sequence=px.colors.qualitative.Set2  # Set the color palette to Set2
    )
    st.plotly_chart(fig)

    # Loan Status Distribution for Each Loan Intent
    st.subheader("Loan Status Distribution for Each Loan Intent")

    # Calculate percentages for pie charts
    percentages = df.groupby(["loan_intent", "loan_status"])["loan_status"] \
        .count() \
        .groupby(level=0) \
        .apply(lambda x: 100 * x / x.sum())

    loan_intents = percentages.index.get_level_values(0).unique()

    # Split loan intents into two groups (for two rows of 3 charts each)
    first_row_intents = loan_intents[:3]
    second_row_intents = loan_intents[3:]

    # Create the first row of charts (3 columns)
    cols1 = st.columns(3)  # Three columns for the first row

# Use the 'Set2' color palette, which has 8 colors by default
    colors = plt.get_cmap('Set2').colors

    for i, intent in enumerate(first_row_intents):
        with cols1[i]:
            fig, ax = plt.subplots(figsize=(4, 4))  # Adjust size for consistency
            
            # Extract the data for the current intent
            intent_percentage = percentages.xs(key=intent, level=0)
            
            # Ensure the colors repeat or scale with the number of categories
            num_colors = len(intent_percentage)
            # Extend the colors list if the number of categories exceeds the available colors
            extended_colors = colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)]

            # Create the pie chart
            wedges, texts, autotexts = ax.pie(
                intent_percentage,
                labels=intent_percentage.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=extended_colors  # Use the extended color list
            )
            
            # Set title and show the pie chart
            ax.set_title(f"Loan Status Distribution for {intent}")
            
            # Show the pie chart in Streamlit
            st.pyplot(fig)

    # Create the second row of charts (3 columns)
    cols2 = st.columns(3)  # Three columns for the second row
    for i, intent in enumerate(second_row_intents):
        with cols2[i]:
            fig, ax = plt.subplots(figsize=(4, 4))  # Adjust size for consistency
            intent_percentage = percentages.xs(key=intent, level=0)
            wedges, texts, autotexts = ax.pie(
                intent_percentage,
                labels=intent_percentage.index,
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title(f"Loan Status Distribution for {intent}")
            st.pyplot(fig)
    median_income_by_grade = df.groupby(['loan_grade', 'loan_intent'])['loan_amnt'].median().reset_index()

    # Pivot the data so that 'loan_grade' becomes the index and 'loan_intent' becomes columns
    pivot_data = median_income_by_grade.pivot(index='loan_grade', columns='loan_intent', values='loan_amnt')

    # Streamlit application
    st.title("Median Loan Amount Analysis")
    st.write("This bar chart shows the **median loan amount by loan grade**, separated by loan intent.")

    # Create the bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_data.plot(kind='bar', ax=ax)

    # Customize chart appearance
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Median Loan Amount")
    ax.set_title("Median Loan Amount by Loan Grade (Separated by Loan Intent)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Rotate x-axis labels for readability

    # Adjust layout
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(fig)
    
    grouped_data = df.groupby(['loan_grade', 'loan_status'])['loan_status'].count().reset_index(name='count')

    # Get unique loan grades and statuses
    loan_grades = grouped_data['loan_grade'].unique()
    loan_statuses = grouped_data['loan_status'].unique()

    # Prepare data for plotting
    data = {status: [] for status in loan_statuses}

    for grade in loan_grades:
        grade_data = grouped_data[grouped_data['loan_grade'] == grade]
        for status in loan_statuses:
            count = grade_data.loc[grade_data['loan_status'] == status, 'count']
            data[status].append(count.iloc[0] if not count.empty else 0)

    # Streamlit app
    st.title("Loan Status Distribution by Loan Grade")
    st.write("This chart visualizes the **loan status distribution** for each loan grade in a grouped bar chart format.")

    # Create the grouped bar chart using Matplotlib
    x = np.arange(len(loan_grades))  # Label locations
    width = 0.2  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, status in enumerate(loan_statuses):
        ax.bar(x + i * width, data[status], width, label=status)

    # Customize the chart
    ax.set_title("Loan Status Distribution by Loan Grade (Grouped)")
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Count")
    ax.set_xticks(x + width * (len(loan_statuses) - 1) / 2)
    ax.set_xticklabels(loan_grades)
    ax.legend(title="Loan Status")

    # Adjust layout for better visualization
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(fig)
    
    # Group data by home ownership and loan status
    home_ownership_loan_status = df.groupby(['person_home_ownership', 'loan_status']).size().unstack()

    # Create the grouped bar chart
    st.subheader("Loan Status Distribution by Home Ownership")

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_palette("Set2")
    home_ownership_loan_status.plot(kind='bar', ax=ax)

    # Set labels and title
    ax.set_xlabel('Home Ownership Status')
    ax.set_ylabel('Count')
    ax.set_title('Loan Status Distribution by Home Ownership')

    # Display the chart in Streamlit
    st.pyplot(fig)

    # Income vs Loan Amount
    st.subheader("Scatter Plot of Income vs Loan Amount")
    fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(df['person_income'], df['loan_amnt'], alpha=0.6, edgecolor='k')

    # Titles and labels
    ax.set_title("Scatter Plot of Income vs Loan Amount")
    ax.set_xlabel("Person Income")
    ax.set_ylabel("Loan Amount")

    # Format X-axis to show numbers in thousands (K)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

    # Show the plot
    st.pyplot(fig)

    # Heatmap of Correlation
    st.subheader("Correlation Heatmap")
    corr_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Columns")
    st.pyplot(fig)

elif option == 'ML':
    # Encoding Categorical Columns
    columns_to_encode = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file',
    'Age Level'
]

    # Initialize LabelEncoders for categorical columns
    encoders = {}
    for column in columns_to_encode:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder

    # Features and target variable
    input_features = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income', 'cb_person_default_on_file',
        'cb_person_cred_hist_length', 'Age Level'
    ]
    X = df[input_features].values
    y = df['loan_status'].values  # Replace 'loan_status' with your actual target column

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions and Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Streamlit UI elements
    st.title("Loan Status Prediction")
    st.write("This machine learning model predicts loan status based on the input features.")

    # Display model accuracy
    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Collect user input from the sidebar
    st.sidebar.header("Input Features")
    home_ownership = st.sidebar.selectbox(
        "Home Ownership", encoders['person_home_ownership'].classes_
    )
    loan_intent = st.sidebar.selectbox("Loan Intent", encoders['loan_intent'].classes_)
    loan_grade = st.sidebar.selectbox("Loan Grade", encoders['loan_grade'].classes_)
    default_on_file = st.sidebar.selectbox(
        "Default on File", encoders['cb_person_default_on_file'].classes_
    )
    age_level = st.sidebar.selectbox("Age Level", encoders['Age Level'].classes_)

    # Collect numeric inputs
    age = st.sidebar.number_input("Age", min_value=18, max_value=100)
    income = st.sidebar.number_input("Income", min_value=0, max_value=1000000)
    emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0, max_value=50)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, max_value=1000000)
    interest_rate = st.sidebar.slider("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)
    percent_income = st.sidebar.slider("Loan Percent Income (%)", min_value=0.0, max_value=100.0, step=0.1)
    credit_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50)

    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [interest_rate],
        'loan_percent_income': [percent_income],
        'cb_person_default_on_file': [default_on_file],
        'cb_person_cred_hist_length': [credit_hist_length],
        'Age Level': [age_level]
    })

    # Encode categorical input features
    for column in columns_to_encode:
        if column in user_input.columns:
            try:
                user_input[column] = encoders[column].transform(user_input[column])
            except ValueError:
                st.warning(
                    f"Warning: '{user_input[column][0]}' is an unseen label for column '{column}'. Using a default value."
                )
                # Assign default value (e.g., first known class)
                user_input[column] = encoders[column].transform([encoders[column].classes_[0]])[0]

    # Scale the input data
    user_input_transformed = sc.transform(user_input)

    # Make the prediction
    prediction = model.predict(user_input_transformed)

    # Display prediction result
    if prediction == 1:
        st.subheader("Prediction: Loan Approved")
    else:
        st.subheader("Prediction: Loan Denied")