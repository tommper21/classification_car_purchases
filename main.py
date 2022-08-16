import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title('Classification Car Purchases')

@st.cache
def load_data():
    data = pd.read_csv('data/car_data.csv')
    rawdata = pd.read_csv('data/car_data.csv')
    data = pd.concat([data,pd.get_dummies(data.Gender)],axis=1)
    data = data.drop(['Gender','User ID'],axis=1)
    y = data.pop('Purchased')
    return rawdata, data, y
rawdata, data, y = load_data()

with st.echo(code_location="below"):
    if st.checkbox('Show raw data'):
        st.write(rawdata)

    with st.sidebar:
        age = st.slider('Age',0,100,20)
        salary = st.slider('Salary',0,200000,20000)
        radio = st.radio("Gender",("Male", "Female"))
        ismale = 0 
        isfemale = 0
        if radio == "Male":
            ismale = 1
        else:
            isfemale = 1
        

    X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    if knn.predict([[age,salary,isfemale,ismale]])[0]:
        st.write("KNN predicts car purchase!")
    else:
        st.write("KNN does not predict car purchase!")
