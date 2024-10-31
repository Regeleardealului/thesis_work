import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.graph_objs as go
import pickle
import time 
import zipfile
import os
from sklearn.decomposition import PCA
import gzip

# Unzipping uploaded files
def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_to)

# Unzip cleaned_usa_accidents.zip and df.zip if they exist
if not os.path.exists('cleaned_usa_accidents.csv'):
    unzip_file('cleaned_usa_accidents.zip', '.')
if not os.path.exists('df.csv'):
    unzip_file('df.zip', '.')

# Set Streamlit page config
st.set_page_config(
    page_title='Car Accident Severity Predictor',
    page_icon='emergency-sign.png',
    layout='wide'
) 

# Loading the dataset
data = pd.read_csv('cleaned_usa_accidents.csv')
data['Start_Time'] = pd.to_datetime(data['Start_Time'])

df = pd.read_csv('df.csv')

# Classification Report Loading section
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

classification_rep_lr = load_pickle('pickle_files/lr_classification_report.pkl')
classification_rep_rf = load_pickle('pickle_files/rf_classification_report.pkl')
classification_rep_bayes = load_pickle('pickle_files/bayes_classification_report.pkl')
classification_rep_nn = load_pickle('pickle_files/classification_report_nn.pkl')

# Confusion Matrix Loading section
conf_matrix_lr = load_pickle('pickle_files/lr_confusion_matrix.pkl')
conf_matrix_rf = load_pickle('pickle_files/rf_confusion_matrix.pkl')
conf_matrix_bayes = load_pickle('pickle_files/bayes_confusion_matrix.pkl')
conf_matrix_nn = load_pickle('pickle_files/confusion_matrix_nn.pkl')

# TOP 10 Features Loading section
top_features_df_lr = load_pickle('pickle_files/feature_importance_lr.pkl')
top_features_df_rf = load_pickle('pickle_files/feature_importance_rf.pkl')
top_features_df_bayes = load_pickle('pickle_files/feature_importance_bayes.pkl')
top_features_df_nn = load_pickle('pickle_files/feature_importance_nn.pkl')

# Loading the encoding Random Forest saved model 
# try:
#     with gzip.open('pickle_files/compressed_random_forest_classifier.pkl.gz', 'rb') as f:
#         model = pickle.load(f)
# except Exception as e:
#     st.error(f"Error loading compressed model: {e}")
#     model = None

with open('pickle_files/random_forest_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load frequency encoding mappings
frequency_mappings = load_pickle('pickle_files/freq_encoding_mappings.pkl')

# Define the function for applying frequency encoding
def apply_frequency_encoding(input_df, mappings, columns):
    for column in columns:
        input_df[column] = input_df[column].map(mappings[column])
        input_df[column].fillna(0, inplace=True)  # Handle unseen categories
    return input_df

# -----------------------------------------------------------------------------------------------------------------------------------
# Streamlit App Framework
st.title("Welcome to the USA Car Accidents Prediction Tool!")

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Visualization", "Model Evaluation", "Prediction"])

with tab1:
    st.header("Few words about the project :thought_balloon:")
    st.write(f"My initial dataset had more than 7 million records, on which I performed a representative sampling to make it more easily handleable and I managed to reduce it to 500 000 records. After some preprocessing steps (like: addressing the missing values, removing outliers, removing irrelevant features, converting data types etc.) my dataset has {data.shape[0]} records with {data.shape[1]} features.")
    st.write("This application aims to predict the `severity` of car accidents on a scale from 1 to 4, where 1 indicates vehicle damage with no serious injuries, 2 represents minor injuries like wounds, 3 involves more serious injuries such as broken bones, and 4 corresponds to severe, life-threatening conditions requiring emergency hospitalization.")
    
    st.write("Here are the first 5 rows of the dataset:")
    st.dataframe(data.head()) 

    st.markdown("<hr>", unsafe_allow_html=True)

    st.image('plots/spearman_corr_matrix.png')

with tab2:
    st.header("Exploratory Data Analysis")
    selection = st.radio(label='Select Visualization Type', options=['Univariate-Bivariate Analysis', 'Map Visualization', 'Time Analysis', 'Dimension Reduction'])  

    if selection == 'Univariate-Bivariate Analysis':
        st.image('plots/accident_by_severity.jpg')
        st.image('plots/proportion_ny_rest_usa.jpg')
        st.image('plots/sunrise_sunset.jpg')
        st.image('plots/poi.jpg')
        st.image('plots/top5_weather.jpg')
        st.image('plots/street_accident.jpg')
    
        # Treemap
        state_accidents = data['State'].value_counts().reset_index()
        state_accidents.columns = ['State', 'Accident Count']
        
        fig = px.treemap(state_accidents, path=['State'], values='Accident Count', title='Accidents by State')
        st.plotly_chart(fig)

        # Funnel Chart
        accidents_by_city = data['City'].value_counts()
        accidents_by_city = accidents_by_city.sort_values(ascending=False)
        top_10_cities = accidents_by_city.head(10)
        
        total_accidents = top_10_cities.sum()
        accidents_percentage = (top_10_cities / total_accidents) * 100
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#C6DCBA', '#bcbd22', '#4793AF']
        
        fig_city = go.Figure()

        fig_city.add_trace(go.Funnel(
            y=top_10_cities.index,
            x=accidents_percentage.values,
            textinfo="percent total",
            marker=dict(color=colors),
            hoverinfo="y+x",
            opacity=0.7,
            connector=dict(line=dict(color="grey", width=1)),
            name="",
        ))

        fig_city.update_layout(
            title='Top 10 Cities with Most Accidents',
            title_x=0.5,  
            xaxis_title='Percentage of Accidents',
            yaxis_title='Cities',
            yaxis=dict(automargin=True),
            plot_bgcolor='rgba(255, 255, 255, 0.7)',  
            paper_bgcolor='rgba(255, 255, 255, 0.7)', 
            font=dict(family="Arial, sans-serif", size=12, color="black"),
            hoverlabel=dict(font=dict(family="Arial, sans-serif", size=14)),  
        )

        st.plotly_chart(fig_city)

        # Counties Chart
        accidents_by_county = data['County'].value_counts()
        accidents_by_county = accidents_by_county.reset_index()
        accidents_by_county.columns = ['County', 'Accident Count']
        accidents_by_county = accidents_by_county.sort_values(by='Accident Count', ascending=False)
        
        fig_county = px.bar(accidents_by_county.head(20), 
                           x='County', 
                           y='Accident Count', 
                           title='Top 20 Counties with Most Accidents', 
                           color='County',
                           color_discrete_sequence=px.colors.qualitative.Pastel1)
        
        fig_county.update_layout(
            xaxis_title='County',
            yaxis_title='Number of Accidents',
            title_x=0.5,  
            title_font=dict(size=20), 
            plot_bgcolor='rgba(255, 255, 255, 0.7)',  
            paper_bgcolor='rgba(255, 255, 255, 0.7)',  
            font=dict(family="Arial, sans-serif", size=12, color="black"),  
            hoverlabel=dict(font=dict(family="Arial, sans-serif", size=14)), 
            showlegend=False
        )
        
        st.plotly_chart(fig_county)

        st.image('plots/ecdf_distance.jpg')
        st.image('plots/ecdf_duration.jpg')
        st.image('plots/rainy.jpg')
        st.image('plots/foggy.jpg')
        st.image('plots/cloudy.jpg')
        st.image('plots/snowy.jpg')
        st.image('plots/windy.jpg')
        st.image('plots/dusty.jpg')
        st.image('plots/thunderstorm.jpg')
        st.image('plots/pressure.jpg')
        st.image('plots/wind_speed.jpg')
        st.image('plots/temperature.jpg')
        st.image('plots/humidity.jpg')
    
    if selection == 'Map Visualization':
        # Map nr.1
        state_accident_counts = pd.value_counts(data['State'])
        map_fig = go.Figure(data=go.Choropleth(
            locations=state_accident_counts.index,
            z=state_accident_counts.values.astype(float),
            locationmode='USA-states',
            colorscale='Viridis', 
            colorbar_title="Number of Accidents",
        ))

        map_fig.update_layout(
            title_text='Accidents by State over the years',
            geo_scope='usa', 
        )

        st.plotly_chart(map_fig)

         # Map nr.2
        data = data.rename(columns={'Start_Lat': 'lat', 'Start_Lng': 'lon'})
        st.map(data[['lat', 'lon']])

        # Map nr.3
        st.image('plots/severity_by_map.jpg')

    if selection == 'Time Analysis':
        st.image('plots/severity_by_day.jpg')
        st.image('plots/severity_by_hour.jpg')
        st.image('plots/severity_by_month.jpg')
        st.image('plots/severity_by_year.jpg')
       
    if selection == 'Dimension Reduction':
        
        X_features = df.loc[:, df.columns != 'Severity']
        target = df['Severity']

        pca = PCA(n_components=8)
        principalComponents = pca.fit_transform(X_features)
        
        # Create a DataFrame for PCA components
        reduced_df = pd.DataFrame(data=principalComponents, columns=[f'PCA{i+1}' for i in range(8)])
        reduced_df['Severity'] = target.values
        
        st.write("PCA Completed. The DataFrame with PCA components is displayed below:")
        st.dataframe(reduced_df)     

        st.image('plots/umap.jpg')
        st.image('plots/pca.jpg')  

with tab3:
    st.header("Model Comparison")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### Logistic Regression")
        st.metric(label='Accuracy', value="60 %", delta="35 %")

    with col2:
        st.markdown("#### Random Forest Classifier")
        st.metric(label='Accuracy', value="93 %", delta="68 %")

    with col3:
        st.markdown("#### Bernoulli Naive Bayes")
        st.metric(label='Accuracy', value="54 %", delta="29 %")

    with col4:
        st.markdown("#### Neural Network")
        st.metric(label='Accuracy', value="86 %", delta="61 %")

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    if st.checkbox("Details for Logistic Regression"):
        st.markdown("### Logistic Regression Classification Report")
        classification_df_lr = pd.DataFrame(classification_rep_lr).transpose()
        st.dataframe(classification_df_lr)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='flare',
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.title('Logistic Regression Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features_df_lr['Feature'], top_features_df_lr['Coefficient'], color='skyblue')
        plt.xlabel('Coefficient')
        plt.title('Top 10 Feature Importance in Logistic Regression')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

    if st.checkbox("Details for Random Forest Classifier"):
        st.markdown("#### Random Forest Classification Report")
        classification_df_rf = pd.DataFrame(classification_rep_rf).transpose()
        st.dataframe(classification_df_rf)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features_df_rf['Feature'], top_features_df_rf['Importance'], color='lightgreen')
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance in Random Forest')
        plt.gca().invert_yaxis()
        st.pyplot(plt)
    
    if st.checkbox("Details for Bernoulli Naive Bayes"):
        st.markdown("#### Bernoulli Naive Bayes Classification Report")
        classification_df_bayes = pd.DataFrame(classification_rep_bayes).transpose()
        st.dataframe(classification_df_bayes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_bayes, annot=True, fmt='d', cmap='viridis',
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.title('Bayes Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features_df_bayes['Feature'], top_features_df_bayes['Coefficient'], color='lightpink')
        plt.xlabel('Coefficient')
        plt.title('Top 10 Feature Importance in Bernoulli Naive Bayes')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

    if st.checkbox("Details for Neural Network"):
        st.markdown("#### Neural Network Classification Report")
        classification_df_nn = pd.DataFrame(classification_rep_nn).transpose()
        st.dataframe(classification_df_nn)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_nn, annot=True, fmt='d', cmap='PiYG',
                    xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.title('Neural Network Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        st.write("#### Top 10 Features for Neural Network")
        st.dataframe(top_features_df_nn)

with tab4:
    st.header('Prediction')
    st.write("Use the form below to enter details about an accident and predict its severity.")

    unique_streets = data['Street'].unique()
    unique_cities = data['City'].unique()
    unique_states = data['State'].unique()
    unique_counties = data['County'].unique()

    with st.form(key='prediction_form'):
        st.write("#### *Do you want to find out how severe is the accident* :question:") 

        distance = st.slider("How long is the estimated queue caused by the accident (in miles)?", min_value=0.0, max_value=100.0)
        county = st.selectbox("Select a county where the accident took place!", options=unique_counties)
        city = st.selectbox("Select a city where the accident took place", options=unique_cities)
        street = st.selectbox("Select a street where the accident took place!", options=unique_streets)
        pressure = st.number_input("Enter the pressure:", min_value=0.0)  
        temperature = st.number_input("Enter the local temperature (in °F):", min_value=0.0)
        humidity = st.number_input("Enter the local humidity:", min_value=0.0)
        state = st.selectbox("Select a state where the accident took place!", options=unique_states)
        wind_speed = st.number_input("Enter the local wind speed:", min_value=0.0)

        # Creating a dataframe based on user inputs
        input_data = pd.DataFrame({
            'Distance(mi)': [distance],
            'County': [county],
            'City': [city],
            'Street': [street],
            'Pressure(in)': [pressure],
            'State': [state],
            'Temperature(F)': [temperature],
            'Wind_Speed(mph)': [wind_speed],
            'Humidity(%)': [humidity]
        })

        # High cardinality columns to apply frequency encoding
        high_cardinality_columns = ['Street', 'City', 'County', 'State']

        # Apply frequency encoding to user input data
        input_data = apply_frequency_encoding(input_data, frequency_mappings, high_cardinality_columns)

        # Submit button
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.03)
                progress_bar.progress(percent_complete + 1)

            # Make prediction on user input values using Random Forest
            prediction = model.predict(input_data)

            if prediction == 1:
                st.write('These inputs indicate vehicle damage with no serious injuries.')
            elif prediction == 2:
                st.write('These inputs indicate minor injuries like wounds.')
            elif prediction == 3:
                st.write('These inputs indicate serious injuries.')
            elif prediction == 4:
                st.write('These inputs indicate severe, life-threatening conditions.')
