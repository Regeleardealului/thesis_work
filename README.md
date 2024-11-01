# USA Car Accident Severity Prediction Tool

Welcome to my thesis project repository! This project focuses on analyzing and predicting the severity of car accidents across the USA based on a variety of factors. The goal was to gain insights into accident patterns and to develop a predictive tool that can estimate the severity of an accident based on several features.
- 📊 Project Overview

  **Dataset:** I used a large dataset containing over 7 million records of car accident data from across the USA. Each record includes a variety of features such as:
  1. Location: start latitude, start longitude, city, county, and street
  2. Time: date and time of the accident
  3. Severity level: ranging from 1 (least severe) to 4 (most severe)
  4. Weather and Road Conditions: temperature, humidity, wind speed, weather conditions, etc.

  **Sampling:** Due to the size of the dataset, I performed a stratified representative sampling to reduce it to 500,000 records. This made it more manageable for processing on my laptop, ensuring a balanced representation while keeping computational requirements in check.

  **Data Preprocessing:** As part of the data cleaning process, I:
  1. Handled missing values and dropped irrelevant features
  2. Created new features to enhance the model's predictive power
  3. Corrected data entry errors and removed outliers
  4. Performed additional standard preprocessing steps to prepare the data for analysis and modeling

  **Exploratory Data Analysis (EDA):** I conducted a thorough EDA to uncover hidden patterns, understand feature distributions, and examine correlations, providing valuable insights into the factors influencing accident severity.

  **Model Development:** I tested several machine learning algorithms to predict the severity level of an accident:
  1. Logistic Regression
  2. Random Forest
  3. Bernoulli Naive Bayes
  4. Neural Network

  📝 After evaluating each model, Random Forest stood out with an accuracy score of 93%.

  **Model Validation:** I further validated the Random Forest model through cross-validation and analyzed its performance using the ROC-AUC metric. The model showed stable performance, reinforcing its reliability for predicting accident severity.

  **Dimensionality Reduction:** Using PCA and UMAP, I visualized the high-dimensional data in 2D and 3D. This helped in understanding the distribution of accident severity levels in lower dimensions.

  **Feature Selection & Deployment Preparation:** For deployment, I reduced the dataset's dimensionality using SelectKBest to retain only the most relevant features. I then rebuilt the Random Forest model using these selected features, optimizing it for faster performance in the web app.

  **Saved Models and Reports:** I saved the correlation matrix, classification reports, and other important analyses as pickle files for easy access. The final model is stored as a joblib file for deployment.

  **Web Application:** The user-friendly GUI for the prediction tool was developed using Streamlit. You can explore and interact with the app here: 👉 (https://usa-accident-severity-prediction.streamlit.app/)

🎓 Learning and Takeaways

Working on this project has been an incredible learning experience! Through each phase, I gained practical knowledge in data handling, modeling, and deployment. I have always been fascinated by the factors influencing accidents, and this project allowed me to explore these elements in depth, ultimately creating a tool that I hope can be insightful and useful.
