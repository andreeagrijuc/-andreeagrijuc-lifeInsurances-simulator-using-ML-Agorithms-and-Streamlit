import streamlit as st
from alg import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analiza():
	st.subheader('Analiza exploratorie a dataset-ului')
	st.write("Continutul acestei pagini reprezinta prelucrarea unui dataset continand 59381 inregistrari anonimizate, "
	         "din viata reala, fiecare avand 17 caracteristici (variabile independente). Astfel, am prezentat prelucrarea dataset-ului, reprezentarea grafica "
	         "a unor variabile in raport cu rezultatul obtinut si, in ultima parte, rezultatele obtinute in urma rularii algoritmilor "
	         "de invatare automata, cuantificate prin metrici de performanta, precum: acuratete, precizie, scorul f, durata de rulare.")
	dataset_intitial = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\dataset_analiza.csv')
	#st.markdown("Numarul de atribute si inregistrari")

	#st.write(dataset.shape)
	st.write("    ")
	st.markdown("Ponderea rezultatelor din dataset-ul neprelucrat")
	def countPlot_init():
		fig = plt.figure(figsize=(10, 4))
		sns.countplot(x="Response", data=dataset_intitial)
		st.pyplot(fig)
	countPlot_init()
	st.markdown("Linia de cod pentru echilibrarea dataset-ului: ")
	response_code = '''dataset['Modified_Response'] = dataset['Response'].apply(
		lambda x: 0 if x <=6  and x >= 0 else (1 if x == 8 or x==7 else -1))'''
	st.code(response_code, language='python')

	st.write("    ")
	st.markdown("Ponderea rezultatelor dupa grupare")
	def countPlot_modif():
		fig = plt.figure(figsize=(10, 4))
		sns.countplot(x="Modified_Response", data=dataset)
		st.pyplot(fig)
	countPlot_modif()
	st.write("    ")
	st.write("Linia de cod pentru aflarea procentului valorilor lipsă din dataset, pe fiecare coloană:")
	miss_vals_code = '''missing_val_count_by_column = dataset.isnull().sum()/len(dataset)'''
	st.code(miss_vals_code, language='python')
	st.write("Graficul valorilor lipsă din dataset")
	def find_miss_vals():
		fig = plt.figure(figsize=(10, 6))
		sns.heatmap(dataset.isna().transpose(), cmap="YlGnBu",cbar_kws={'label': 'Missing Data'})
		st.pyplot(fig)
	find_miss_vals()
	st.write("Metoda pentru reprezentarea grafică a valorilor lipsă:")
	miss_vals = '''	def find_miss_vals():
		fig = plt.figure(figsize=(10, 6))
		sns.heatmap(dataset.isna().transpose(), cmap="YlGnBu",cbar_kws={'label': 'Missing Data'})
		st.pyplot(fig)'''
	st.code(miss_vals, language='python')
	#percent_missing = dataset.isnull().sum() * 100 / len(dataset)
	#missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
	#st.write(missing_value_df)
	st.write("Am înlocuit valorile lipsă cu media acestora, prin următoarea linie de cod, unde X reprezintă variabilele independente în funcție de care se returnează rezultatele : ")
	miss_vals = '''	X = X.fillna(X.mean())'''
	st.code(miss_vals, language='python')

	st.write("Linia de cod prin care se grupează dataset-ul în parte de antrenare, cu procentaj de 80% și parte de test, cu procentaj de 20%: ")
	miss_vals = '''X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=1)'''
	st.code(miss_vals, language='python')
	st.write("    ")
	st.write("    ")


	st.markdown("Se poate observa faptul că Indicele de masă corporală are valori mai mici in cazul scorului 1")
	def boxplot_BMI():
		fig = plt.figure(figsize=(10, 8))
		sns.set_theme(style="whitegrid")
		sns.boxplot(x="Modified_Response", y="BMI", data=dataset)
		st.pyplot(fig)
	boxplot_BMI()
	st.write("Metoda pentru reprezentarea grafică a indicelui de masă corporală:")
	bmi_plot_code = '''	def boxplot_BMI():
		fig = plt.figure(figsize=(10, 8))
		sns.set_theme(style="whitegrid")
		sns.boxplot(x="Modified_Response", y="BMI", data=dataset)
		st.pyplot(fig)'''
	st.code(bmi_plot_code, language='python')
	st.write("    ")
	st.markdown("De asemenea, persoanele mai in varstă tind să primească raspunsul 0")
	def boxplot_Age():
		fig = plt.figure(figsize=(10, 8))
		sns.set_theme(style="whitegrid")
		sns.boxplot(x="Modified_Response", y="Ins_Age", data=dataset, palette="Set2")
		st.pyplot(fig)
	boxplot_Age()
	st.write("Metoda pentru reprezentarea grafică a vârstei:")
	age_plot_code = '''	def boxplot_Age():
		fig = plt.figure(figsize=(10, 8))
		sns.set_theme(style="whitegrid")
		sns.boxplot(x="Modified_Response", y="Ins_Age", data=dataset, palette="Set2")
		st.pyplot(fig)'''
	st.code(age_plot_code, language='python')
	st.write("    ")
	st.markdown("Rezultatele obținute")
	rez_obtinute = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	df = pd.DataFrame(rez_obtinute)
	st.dataframe(df)