import os
import streamlit as st
from main import Engine

predictor = Engine()
uploaded_file = st.file_uploader(label = "Upload image")
temp_path = os.path.join("data/temp",uploaded_file.name)
if uploaded_file is not None:
    with open(temp_path,"wb") as f:
         f.write(uploaded_file.getbuffer())    
    _, col22, _ = st.columns([1,2, 1])
    _, col21, _ = st.columns([0.1,1, 0.1])
    col22.image(uploaded_file, use_column_width=True)
    kind, result = predictor.run(os.path.join("data/temp",uploaded_file.name))
    if kind == "DOG":
        col21.markdown("<h2 style='text-align: center; color: green;'>This dog is {} breed.</h2>".format(result), unsafe_allow_html=True)
    elif kind == "HUMAN":
        col21.markdown("<h2 style='text-align: center; color: blue;'>The resembling dog breed is {}</h2>".format(result), unsafe_allow_html=True)
    else:
        col21.markdown("<h2 style='text-align: center; color: red;'>This image does not have any dog or human.</h2>", unsafe_allow_html=True)
    os.remove(temp_path)