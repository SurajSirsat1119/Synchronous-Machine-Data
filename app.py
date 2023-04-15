import streamlit as st
import pickle

import numpy as np


st.title('Synchorouns Machine')

number1 = st.number_input('Load Current')
st.write('load Current ', number1)

number2 = st.number_input('Power Factor')
st.write('Power Factor ', number2)

number3 = st.number_input('Power Factor error')
st.write('Power factor error', number3)

number4 = st.number_input('Changing of excitation current of synchronous machine')
st.write('Changing of excitation current of synchronous machine ', number4)

##data = pickle.load(open('synchronous.pkl','rb'))
##scale = pickle.load(open('scale.pkl','rb'))
##model = pickle.load(open('model.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))
##y_train = pickle.load(open('y_train.pkl','rb'))
##X_train =pickle.load(open('X_train.pkl','rb'))
if st.button('Predication'):
    query = [[number1,number2,number3,number4]]
    st.title(pipe.predict(query))



