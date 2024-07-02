import pandas as pd
import numpy as np
import streamlit as st
import pickle

st.header("Banglore House price predictor")

df=pickle.load(open("data_house.pkl",'rb'))
pipe=pickle.load(open("pipe_houseprice1.pkl",'rb'))

area=st.selectbox('Location',df['location'].unique())
bhk=st.number_input('bhk')
sqft=st.number_input('Total sqft')
bath=st.number_input('bathrom')
balcony=st.number_input('Balcony')

if st.button('Predict'):
    dict={'location':{0:area},'size':{0:bhk},'total_sqft':{0:sqft},'bath':{0:bath},'balcony':{0:balcony}}
    d=pd.DataFrame(dict)
    st.title(int(np.exp(pipe.predict(d)[0])*100000))



