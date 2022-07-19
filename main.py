import streamlit as st
from polite import polite
from home import home
from data_input import data_input
from analiza import analiza
from database import *
from PIL import Image

def main():
	#st.title("Smart Insurances")

	image = Image.open('logo smart insurances.jpg')

	st.image(image)
	with st.sidebar:
		st.subheader("Bine ai venit, {} !".format(st.session_state.prenume_user))
	menu = ["Home", "Gestiune Polite", "Simulare","Analiza Exploratorie"]
	choice = st.sidebar.selectbox("Meniu", menu)

	if choice == "Analiza Exploratorie":
		analiza()
	elif choice == "Gestiune Polite":
		polite()
	elif choice == "Home":
		home()
	elif choice == "Simulare":
		data_input()


if __name__ == "__main__":
	main()
