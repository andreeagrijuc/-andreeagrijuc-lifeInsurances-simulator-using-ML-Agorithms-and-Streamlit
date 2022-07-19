import streamlit as st
from PIL import Image

def home():
	st.write(" ")
	st.subheader("Evaluarea profilurilor de risc tocmai a devenit mai ușoară, cu ajutorul tehnologiilor din Machine Learning ")

	image = Image.open('home4.jpg')

	st.image(image)


