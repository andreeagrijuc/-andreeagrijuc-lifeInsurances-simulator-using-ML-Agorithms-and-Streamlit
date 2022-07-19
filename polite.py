import pandas as pd
import streamlit as st
from database import *


def polite():
	st.subheader("Gestiunea polițelor")
	def incarca_tabel_polite():
		result_p = view_polite()
		df = pd.DataFrame.from_records(result_p,
		                               columns=['Nume Asigurat', 'Prenume Asigurat', 'Serie Polita', 'Nr Polita',
		                                        'Suma Asigurata', 'Prima Asigurata', 'Data Inceput', 'Data Sfarsit',
		                                        'idUser'])
		filtered_data = df[df['idUser'] == st.session_state['idUser']]
		return filtered_data

	meniu_p=["Adaugă polițe", "Vizualizează polițe", "Editează polițe", "Șterge polițe"]
	choice_p=st.selectbox("Va rugăm selectati:",meniu_p)
	if choice_p=="Vizualizează polițe":
		result = view_polite()
		df = pd.DataFrame.from_records(result,
		                               columns=['Nume Asigurat', 'Prenume Asigurat', 'Serie Polita', 'Nr Polita',
		                                        'Suma Asigurata', 'Prima Asigurata', 'Data Inceput', 'Data Sfarsit',
		                                        'idUser'])
		filtered_data = df[df['idUser'] == st.session_state['idUser']]
		if filtered_data.empty:
			st.write("Nu exista polite salvate ale utilizatorului {}".format(st.session_state['emailUser']))
		else:
			st.dataframe(filtered_data)

	elif choice_p=="Adaugă polițe":
		with st.form(key="AdaugaPolite"):
			st.subheader("Introduceți datele:")
			col1, col2 = st.columns(2)
			with col1:
				nume_asigurat = st.text_input("Nume Asigurat")
				valabilitate_inceput=st.date_input("Data Inceput")
				serie_polita=st.text_input("Serie polita")
				suma_asig=st.number_input("Suma asigurata")
				suma_asig.__round__(0)
			with col2:
				prenume_asigurat = st.text_input("Prenume Asigurat")
				valabilitate_sfarsit = st.date_input("Data Sfarsit")
				nr_polita = st.text_input("Numar polita")
				prima_asig = st.number_input("Prima asigurata")
				prima_asig.__round__(0)
			submit_btn = st.form_submit_button(label="Salveaza polita")
			if submit_btn:
				if nume_asigurat != '' and prenume_asigurat != '' and serie_polita != '' and nr_polita != '' and suma_asig != '' and prima_asig != '':
					if serie_polita.isnumeric() == True:
						st.error("Seria poliței trebuie sa conțina numai litere")
					if nr_polita.isnumeric() == False or suma_asig.is_integer() == False or prima_asig.is_integer() == False:
						st.error("Format invalid pentru numărul poliței, suma asigurată sau prima de asigurare.")
					else:
						save_date_polita(nume_asigurat, prenume_asigurat, valabilitate_inceput, valabilitate_sfarsit, serie_polita, nr_polita, suma_asig, prima_asig, st.session_state['idUser'])
						st.success("Datele poliței cu seria {} si numarul {} au fost salvate".format(serie_polita, nr_polita))
				else:
					st.error("Va rugăm să introduceți toate datele poliței")

	elif choice_p == "Editează polițe":
		idUser = st.session_state['idUser']
		lista_polite=view_serie_nr_polite(idUser)
		if len(lista_polite) != 0:

			st.dataframe(incarca_tabel_polite())
			select_polita=st.selectbox("Selectati polita:", lista_polite)
			result=get_polita_by_nr(select_polita[0], select_polita[1])
			ret_nume=result[0][0]
			ret_prenume = result[0][1]
			ret_serie=result[0][2]
			ret_nr=result[0][3]
			ret_suma_asig=result[0][4]
			ret_prima=result[0][5]
			ret_data_inc=result[0][6]
			ret_data_sf=result[0][7]
			if select_polita:
				with st.form(key="EditeazaPolite"):
					st.subheader("Editati polita")
					col1, col2 = st.columns(2)
					with col1:
						edit_nume_asigurat = st.text_input("Nume Asigurat",ret_nume)
						edit_valabilitate_inceput = st.date_input("Data Inceput", ret_data_inc)
						edit_serie_polita = st.text_input("Serie polita", ret_serie, disabled=True)
						edit_suma_asig = st.number_input("Suma asigurata", ret_suma_asig)
					with col2:
						edit_prenume_asigurat = st.text_input("Preume Asigurat", ret_prenume)
						edit_valabilitate_sfarsit = st.date_input("Data Sfarsit", ret_data_sf)
						edit_nr_polita = st.text_input("Numar polita", ret_nr, disabled=True)
						edit_prima_asig = st.number_input("Prima asigurata", ret_prima)
					st.warning("Salvați modificarile poliței ?")
					submit_btn = st.form_submit_button(label="Salveaza modificarile")

					if submit_btn:
						edit_date_polita(edit_nume_asigurat, edit_prenume_asigurat, edit_valabilitate_inceput, edit_valabilitate_sfarsit,
						                 edit_serie_polita, edit_suma_asig, edit_prima_asig, edit_nr_polita)
						st.success("Datele politei au fost editate")
						incarca_tabel_polite()
		else:
			st.write("Nu exista polite salvate ale utilizatorului {}".format(st.session_state['emailUser']))

	elif choice_p == "Șterge polițe":
		idUser = st.session_state['idUser']
		lista_polite = view_serie_nr_polite(idUser)
		if len(lista_polite) != 0:
			st.dataframe(incarca_tabel_polite())

			select_polita = st.selectbox("polite", lista_polite)
			result = get_polita_by_nr(select_polita[0], select_polita[1])
			delete_btn=st.button("Șterge polita selectata")
			if delete_btn:
				delete_polita(select_polita[0], select_polita[1])
				st.warning("Polita cu seria: {} si numarul: {} a fost stearsa".format(select_polita[0], select_polita[1]))
				incarca_tabel_polite()
		else:
			st.write("Nu exista polite salvate ale utilizatorului {}".format(st.session_state['emailUser']))