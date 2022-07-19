import streamlit as st
from database import *
from main import main

def register():
	first_name = st.text_input("First Name", key='fName')
	last_name = st.text_input("Last Name", key='lName')
	new_email = st.text_input("Your Email", key='regEmail')
	new_password = st.text_input("Your Password", type='password', key='regPass')
	register_btn = st.button("Register", key='regBtn')
	#st.write(st.session_state)
	if register_btn:
		hashed_pass=hash_passwords(new_password)
		add_values(new_email, hashed_pass, first_name, last_name)
		st.success("Created the account: {}".format(new_email))
		st.session_state.loggedIn = True
		main()
	else:
		st.error("Something went wrong")
		st.session_state.loggedIn = False

def showRegisterPage():
	with registerSection:
		loginSection.empty()
		first_name = st.text_input("First Name", key='fName')
		last_name = st.text_input("Last Name", key='lName')
		new_email = st.text_input("Your Email", key='regEmail')
		new_password = st.text_input("Your Password", type='password', key='regPass')
		register_btn = st.button("Register", key='regBtn')
		register()


def register():
	#st.write(st.session_state)
	if register_btn:
		hashed_pass=hash_passwords(new_password)
		add_values(new_email, hashed_pass, first_name, last_name)
		st.success("Created the account: {}".format(new_email))
		st.session_state.loggedIn = True
		main()
	else:
		st.error("Something went wrong")
		st.session_state.loggedIn = False

first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
new_email = st.text_input("Your Email")
new_password = st.text_input("Your Password", type='password')
register_btn = st.button("Register")
if register_btn:
    hashed_pass=hash_passwords(new_password)
    add_values(new_email, hashed_pass, first_name, last_name)
    st.success("Created the account: {}".format(new_email))
else:
    st.warning("Something went wrong")

