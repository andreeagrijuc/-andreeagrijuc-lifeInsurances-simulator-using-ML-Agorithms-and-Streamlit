from flask import Flask, abort, redirect, url_for
from database import *
import streamlit as st
from main import main
from PIL import Image


headerSection=st.container()
mainSection=st.container()
loginSection=st.container()
registerSection=st.container()
logoutSection=st.container()
fname_cont=st.empty()

def show_main():
	with mainSection:
		main()

def LoggedOut_Clicked():
	st.session_state['loggedIn'] = False
	for key in st.session_state.keys():
		del st.session_state[key]

def show_logout_page():
	loginSection.empty()
	registerSection.empty()
	with logoutSection:
		with st.sidebar:
			st.button("Log Out", key="logout", on_click=LoggedOut_Clicked)

def LoggedIn_Clicked(email, password):
	if email != '' and password != '':
		hashed_entered_pass = hash_passwords(password)
		account_data = login_user(email, hashed_entered_pass)

		if account_data:
			st.session_state['loggedIn'] = True
			st.success("Logged in as: {}".format(email))
			if 'emailUser' not in st.session_state:
				st.session_state['emailUser'] = email
				st.write(login_user(email, hashed_entered_pass))
				if 'idUser' not in st.session_state:
					st.session_state['idUser'] = account_data[0][0]
				if 'prenume_user' not in st.session_state:
					st.session_state['prenume_user'] = account_data[0][4]
		else:
			st.session_state['loggedIn'] = False
			st.error("Incorrect email / password")

def Register_Clicked(new_email, new_password, first_name, last_name):
	loginSection.empty()
	if new_email != '' and new_password != '':
		hashed_pass = hash_passwords(new_password)
		register_data = add_values(new_email, hashed_pass, first_name, last_name)
		st.session_state['registered'] = False

		if register_data:
			st.success("Created the account: {}".format(new_email))
			st.session_state['registered'] = False
			if 'emailUser' not in st.session_state:
				st.session_state['emailUser'] = new_email
		else:
			st.session_state['registered'] = False

def showLoginPage():
	with fname_cont:
		registerSection.empty()
		with loginSection:
			col1, col2, col3 = st.columns([2, 4, 2])
			with col1:
				st.write("")
			with col2:
				image = Image.open('logo smart insurances.jpg')
				st.image(image)
				st.title(" Autentificare")
			with col1:
				st.write("")
			if st.session_state['loggedIn'] == False:
				registerSection.empty()
				email = st.text_input("Email", key='loginEmail')
				password = st.text_input("Parola", type='password', key='loginPass')
				login_btn = st.button("Log In", key='loginBtn', on_click=LoggedIn_Clicked(email, password))
				if login_btn:
					if email == '':
						st.warning("Va rugam introduceti un email valid")
					elif password == '':
						st.warning("Va rugam introduceti parola")
				st.write("Nu ai un cont?")
				go_to_register_btn=st.button("Mergi la register")
				if go_to_register_btn:
					st.session_state['registered'] = True


def showRegisterPage():

	with registerSection:
		if st.session_state['registered']:
			col1, col2, col3 = st.columns([2, 4, 2])
			with col1:
				st.write("")
			with col2:
				image = Image.open('logo smart insurances.jpg')
				st.image(image)
				st.title("Inregistrare cont")
			with col1:
				st.write("")

			first_name = st.text_input("Prenume", key='fName')
			last_name = st.text_input("Nume", key='lName')
			new_email = st.text_input("Email", key='regEmail')
			new_password = st.text_input("Parola", type='password', key='regPass')
			register_btn = st.button("Register", key='regBtn',on_click=Register_Clicked(new_email, new_password, first_name, last_name))
			btnAux = st.button("Inapoi")
			if btnAux:
				st.session_state["registered"] = False

with headerSection:
	if 'loggedIn' not in st.session_state:
		st.session_state['loggedIn'] = False
		showLoginPage()
	else:
		if 'registered' not in st.session_state:
			st.session_state['registered'] = False
		elif 'registered' in st.session_state and st.session_state['registered']:

			showRegisterPage()

		if st.session_state['loggedIn']:
			show_main()
			show_logout_page()
		else:
				if st.session_state['loggedIn'] == False and st.session_state['registered'] == False:
					showLoginPage()













