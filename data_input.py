import streamlit as st
from database import *
import pandas as pd
import numpy as np
from datetime import date, datetime
from alg import *

def data_input():
	file=f'C:\\Users\\andre\\Desktop\\licenta\\input_dataset.csv'
	input_dt = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\input_dataset.csv')
	def valori_norm(input, media, dev_std):
		min = media - 2 * dev_std
		max = media + 2 * dev_std
		val_norm = (input - min) / (max - min)
		return val_norm
	def valori_scal(input, min, max):
		val_scal = (input - min) / (max - min)
		return val_scal

	with st.form(key="DatePers"):
		st.subheader("Date Personale:")
		col1,col2,col3 = st.columns(3)
		with col1:
			#nume_input = st.text_input("Nume")

			inaltime_input = st.number_input("Inaltime (CM)", value=175, key="inaltime")
			inaltime_input = valori_scal(inaltime_input, 152, 205).__round__(6)

			lista_dom_activ = bring_dom_activ()
			dom_activ_input=st.selectbox("Domeniul de activitate", [i[1] for i in lista_dom_activ])
			scor_domeniu = bring_dom_activ_nume(dom_activ_input)
			scor_domeniu = scor_domeniu[0] / 10

		with col2:
			#prenume_input = st.text_input("Prenume")
			greutate_input = st.number_input("Greutate (KG)", value=75,key="greutate")
			greutate_input = valori_scal(greutate_input, 43, 110).__round__(6)
			lista_hobby=bring_hobbys()
			lista_hobby_input = st.multiselect("Hobby-uri", [i[1] for i in lista_hobby], args=[[i[1] for i in lista_hobby]])
			scor_hobby = 0
			for hobby_sel in lista_hobby_input:
				for hobby in lista_hobby:
					if hobby_sel == hobby[1]:
						scor_hobby = scor_hobby + hobby[2]

		with col3:

			data_nasterii = st.date_input("Data nasterii", date(1990, 1, 1))

			varsta=(date.today().year - data_nasterii.year)

			varsta=valori_scal(varsta,18,90).__round__(6)

			st.write("Indicele dvs de greutate corporala este:")
			bmi2 = st.session_state.greutate / ((st.session_state.inaltime / 100) * (st.session_state.inaltime / 100))
			bmi_afisat=bmi2
			bmi2=valori_scal(bmi2, 15, 45)
			st.write(bmi_afisat.__round__(2))

		st.subheader("Istoric medical:")
		col4, col5, col6 = st.columns(3)
		with col4:
			afect_card_input = st.selectbox(
				"Istoric de afectiuni cardiace? Hiperensiune arteriala, AVC, alte boli coronariene", ["Nu", "Da"])
			afect_card_input = 3 if afect_card_input == 'Nu' else 1

			are_diabet_input = st.selectbox("Diabet? ", ["Nu", "Da"])
			are_diabet_input = 2 if are_diabet_input == 'Nu' else 3

			are_ulcer_input = st.selectbox("Ulcer gastric si/sau duodenal ? ", ["Nu", "Da"])
			are_ulcer_input=0 if are_ulcer_input=='Nu' else 1

			lista_afect_cronice = bring_afect_cronice()
			istoric_fam_af_cronice_input = st.multiselect("Istoric in familie de afectiuni cronice?",[i[1] for i in lista_afect_cronice])
			scor_afect_cronice = 0
			for afect_sel in istoric_fam_af_cronice_input:
				for afect in lista_afect_cronice:
					if afect_sel == afect[1]:
						scor_afect_cronice = scor_afect_cronice + afect[2] / 10

		with col5:
			afect_lomb_input = st.selectbox("Afectiuni lombare sau cervicale? ", ["Nu", "Da"])
			afect_lomb_input = 3 if afect_lomb_input == 'Nu' else 1

			lipide_sange_input = st.selectbox("Nr crescut de lipide in sange? ", ["Nu", "Da"])
			lipide_sange_input = 0 if lipide_sange_input == 'Nu' else 1

			afect_maligne_input = st.selectbox("Suferiti de afectiuni maligne in prezent? ", ["Nu", "Da"])
			afect_maligne_input = 0 if afect_maligne_input == 'Nu' else 1

			lista_afect_maligne = bring_afect_maligne()
			istoric_fam_af_maligne_input = st.multiselect("Istoric in familie de afectiuni maligne?",[i[1] for i in lista_afect_maligne])
			scor_afect_maligne = 0
			for afect_sel in istoric_fam_af_maligne_input:
				for afect in lista_afect_maligne:
					if afect_sel == afect[1]:
						scor_afect_maligne = scor_afect_maligne + afect[2] / 10
		with col6:
			fumator_input = st.selectbox("Sunteti fumator?", ["Nu", "Da"])
			fumator_input = 2 if fumator_input == 'Nu' else 1

			afect_renale_input = st.selectbox("Afectiuni renale cronice? ", ["Nu", "Da"])
			afect_renale_input = 2 if afect_renale_input == 'Nu' else 1

			cons_alcool_input = st.selectbox("Consumati alcool mai des de 2 ori pe saptamana? ", ["Nu", "Da"])
			cons_alcool_input = 0 if cons_alcool_input == 'Nu' else 1



		input_data = {
			'Id':1,
			'Ins_Age': varsta,
			'Ht':inaltime_input,
			'Wt': greutate_input,
			'BMI': bmi2,
			'Employment_Info_2': scor_domeniu,
			'Employment_Info_1':scor_hobby,
			'Family_Hist_3': scor_afect_cronice,
			'Family_Hist_4': scor_afect_maligne,
			'Medical_History_4': fumator_input,
			'Medical_History_13': afect_lomb_input,
			'Medical_History_20': afect_renale_input,
			'Medical_History_23': afect_card_input,
			'Medical_History_30': are_diabet_input,
			'Medical_Keyword_3': lipide_sange_input,
			'Medical_Keyword_15': cons_alcool_input,
			'Medical_Keyword_40': afect_maligne_input,
			'Medical_Keyword_43': are_ulcer_input
	}


		submit_btn = st.form_submit_button(label="Incarca datele")
		if submit_btn:
			#send_insured_info_to_db(nume_input, prenume_input, data_nasterii, inaltime_input, greutate_input, bmi2, scor_domeniu,
			                        #scor_hobby, afect_card_input, are_diabet_input, are_ulcer_input, afect_lomb_input, lipide_sange_input,
			                        #afect_maligne_input, fumator_input, afect_renale_input, cons_alcool_input, scor_afect_cronice, scor_afect_maligne)
			st.success("Submitted")
			input_dt = input_dt.drop(input_dt.index[0])

			input_dt = input_dt.append(input_data, ignore_index=True)
			input_dt.to_csv(file, index=False)
			st.warning("Atenție! Aceste scoruri returnate de algoritmi nu reprezintă un diagnostic specializat și trebuie tratate în scop pur informativ. "
			           "Pentru orice simptome asupra cărora aveți nelămuriri, vă rugăm să vă adresați unui cadru medical specializat.")
			st.info("Valoarea 1 semnifică o stare de sănătate generală foarte bună, iar valoarea 0 indică posibilitatea prezenței anumitor afecțiuni.")

			lr_result=LRegr2()
			st.write('Algoritmul Logistic regression a returnat: ',lr_result)
			rf_result=RandForest2()
			st.write('Algoritmul Random Forest a returnat: ', rf_result)
			svm_result = SVM2()
			st.write('Algoritmul Support Vector Machines a returnat: ', svm_result)


