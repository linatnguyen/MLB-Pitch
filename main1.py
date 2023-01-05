import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import joblib

# make the page wide
st.set_page_config(layout = 'wide')

# create containers for different sections
header = st.container()
eda1 = st.container()
modeling = st.container()

with header:
    st.title('2011 MLB Pitch Prediction')
    st.text('Lina Nguyen')
    df = pd.read_csv('Data/KNN_df.csv')
    eda = pd.read_csv('Data/eda.csv')
    eda = eda.dropna()

    pitch_label = {'SI': 'Sinker', 'CU':'Curveball', 'FF': 'Fastball', 'SL': 'Slider', 'KN': 'Knuckleball', 'FT': 'Two-Seam Fastball', 'CH': 'Changeup', 'FS': 'Splitter', 'KC': 'Knuckle-curve', 'FC': 'Cutter'}
    eda['pitch_type'] = eda['pitch_type'].map(pitch_label)

with eda1:
    st.header('Exploratory Data Analysis')
    st.subheader('Clean 2011 MLB Pitch Dataset')
    st.write(df.head())

    st.subheader('Pitch Distribution')
    pitch_dist = pd.DataFrame(eda['pitch_type'].value_counts())
    st.bar_chart(pitch_dist)

    col1, col2 = st.columns(2)        

    st.subheader('3d Pitch Model')
    opt = st.selectbox('Select Pitch Model', ('Pitch Distance (ft) from Home Plate', 'Velocity (ft/s) of Pitch', 'Acceleration (ft/s/s) of Pitch'))
    if opt == 'Pitch Distance from Home Plate':
        fig = px.scatter_3d(eda, x = 'x0', y = 'y0', z = 'z0', color = 'pitch_type')
        opt1 = st.multiselect('Select Pitch Type', ('Sinker', 'Curveball', 'Fastball', 'Slider', 'Knuckleball', 'Two-Seam Fastball', 'Changeup', 'Splitter', 'Knuckle-curve', 'Cutter'))
        st.plotly_chart(fig)
    elif opt == 'Velocity (ft/s) of Pitch':
        fig11 = px.scatter_3d(eda, x = 'vx0', y = 'vy0', z = 'vz0', color = 'pitch_type')
        st.plotly_chart(fig11)
    else:
        fig22 = px.scatter_3d(eda, x = 'ax', y = 'ay', z = 'az', color = 'pitch_type')
        st.plotly_chart(fig22)

with modeling:
    st.subheader('Simple Pitch Prediction with Random Forest')
    st.write('This model is a simpler version of the model in the notebook. Only 7 predictors were chosen for this model.')
    X = df[['inning', 'balls', 'strikes', 'fouls', 'outs', 'batter_id', 'pitcher_id']]
    y = df['pitch_type']
    rf = RandomForestClassifier()
    rf.fit(X,y)
    joblib.dump(rf, "rf.plk")

    inning = st.selectbox("Inning Number", options = df['inning'].unique())
    balls = st.selectbox('Number of Balls', options = df['balls'].unique())
    strikes = st.selectbox('Number of Strikes', options = df['strikes'].unique())
    fouls = st.selectbox('Number of Fouls', options = df['fouls'].unique())
    outs = st.selectbox('Number of Outs', options = df['outs'].unique())
    batter_id = st.selectbox('Batter Id', options = df['batter_id'].unique())
    pitcher_id = st.selectbox('Pitcher Id', options = df['pitcher_id'].unique())
    
    if st.button("Submit"):
        clf = joblib.load("rf.plk")
        X1 = pd.DataFrame([['inning', 'balls', 'strikes', 'fouls', 'outs', 'batter_id', 'pitcher_id']])
        prediction = clf.predict(X)[0]
        pitch_label1 = {'0': 'Fastball', '1':'Slider', '2': 'Curveball', '3': 'Sinker', '4': 'Cutter', '5': 'Two-Seam Fastball', '6': 'Changeup'}
        y['pitch_type'] = y['pitch_type'].map(pitch_label1)
        st.text(f"The model predicts a {prediction} will be thrown next")






