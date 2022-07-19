import streamlit as st
import pyodbc
import hashlib

def init_connection():
	return pyodbc.connect(
		"DRIVER={ODBC Driver 17 for SQL Server};SERVER=ANDREEA;DATABASE=licenta;UID=andreeatest;PWD=andreea"
	)
conn = init_connection()

@st.experimental_memo(ttl=600)
def login_user(email, passHash):
	with conn.cursor() as cur:
		cur.execute("SELECT * FROM Useri WHERE email=? AND passHash=?;", (email, passHash))
		return cur.fetchall()

def hash_passwords(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_pass_entered):
	if hash_passwords(password) == hashed_pass_entered:
		return hashed_pass_entered
	else:
		return False

def get_users():
	with conn.cursor() as cur:
		cur.execute("SELECT email FROM Useri;")
		for row in cur:
			row_to_list = [elem for elem in row]
		return row_to_list
def get_passwords():
	with conn.cursor() as cur:
		cur.execute("SELECT passHash FROM Useri;")
		for row in cur:
			row_to_list = [elem for elem in row]
		return row_to_list
def get_names():
	with conn.cursor() as cur:
		cur.execute("SELECT prenume FROM Useri;")
		for row in cur:
			row_to_list = [elem for elem in row]
		return row_to_list

def run_query(query):
	with conn.cursor() as cur:
		cur.execute(query)
		return cur.fetchall()

def add_datas(query):
	with conn.cursor() as cur:
		cur.execute(query)
		conn.commit()
# add_to_db=add_datas("INSERT INTO Useri VALUES ('test@test.ro', '234', 'Floricel', 'Test'), ('alttest.ro', '345', 'Ema', 'Test');")

def add_values(email, passHash, nume, prenume):
	with conn.cursor() as cur:
		cur.execute("INSERT INTO Useri VALUES (?, ?, ?, ?);", (email, passHash, nume, prenume))
		conn.commit()
# add_values('new@altceva.ro','123','Miau','Supermiau')


def insert_data(query):
	with conn.cursor() as cur:
		cur.execute(query)
		conn.commit()

def bring_dom_activ():
	with conn.cursor() as cur:
		cur.execute("SELECT * from DomeniiActivitate;")
		return cur.fetchall()
def bring_dom_activ_nume(denumire):
	with conn.cursor() as cur:
		cur.execute('SELECT scor from DomeniiActivitate WHERE denumireDomActivitate=?;',(denumire))
		#return cur.fetchall()
		for row in cur:
			row_to_list = [elem for elem in row]
		return row_to_list

def bring_hobbys():
	with conn.cursor() as cur:
		cur.execute("SELECT * from Hobbyuri;")
		return cur.fetchall()

def bring_afect_cronice():
	with conn.cursor() as cur:
		cur.execute("SELECT * from ListaAfectiuniCronice;")
		return cur.fetchall()

def bring_afect_maligne():
	with conn.cursor() as cur:
		cur.execute("SELECT * from ListaAfectiuniMaligne;")
		return cur.fetchall()

def bmi_calc():
	return st.session_state.greutate / ((st.session_state.inaltime / 100) * (st.session_state.inaltime / 100))

def send_insured_info_to_db(nume_input, prenume_input, data_nasterii, inaltime_input, greutate_input, bmi2, scor_domeniu,
			                        scor_hobby, afect_card_input, are_diabet_input, are_ulcer_input, afect_lomb_input, lipide_sange_input,
			                        afect_maligne_input, fumator_input, afect_renale_input, cons_alcool_input, scor_afect_cronice, scor_afect_maligne):
	with conn.cursor() as cur:
		cur.execute("INSERT INTO Asigurati (nume, prenume, dataNasterii, inaltime, greutate, BMI, scorDomActivitate, "
		            "scorHobby, afectCardiace, areDiabet, areUlcer, afectColoana, lipideInSange,"
		            "afectMaligne, esteFumator, afectRenale, consAlcool, istoricFamAfectCronice, istoricFamAfectMaligne) "
		            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
		            (nume_input, prenume_input, data_nasterii, inaltime_input, greutate_input, bmi2, scor_domeniu,
			                        scor_hobby, afect_card_input, are_diabet_input, are_ulcer_input, afect_lomb_input, lipide_sange_input,
			                        afect_maligne_input, fumator_input, afect_renale_input, cons_alcool_input, scor_afect_cronice, scor_afect_maligne))
		conn.commit()
def save_date_polita(nume_asigurat, prenume_asigurat, valabilitate_inceput, valabilitate_sfarsit, serie_polita, nr_polita, suma_asig, prima_asig, idUser):
	with conn.cursor() as cur:
		cur.execute("INSERT INTO Polite (numeAsigurat, prenumeAsigurat, dataInceput, dataSfarsit, seriePolita, numarPolita, sumaAsigurata, "
		            "primaAsig, idUser) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
		            (nume_asigurat, prenume_asigurat, valabilitate_inceput, valabilitate_sfarsit, serie_polita, nr_polita, suma_asig, prima_asig, idUser))
		conn.commit()

def edit_date_polita(nume_asigurat, prenume_asigurat, valabilitate_inceput, valabilitate_sfarsit, serie_polita, suma_asig, prima_asig, nr_polita):
	with conn.cursor() as cur:
		cur.execute("UPDATE Polite "
		            "SET numeAsigurat=?, prenumeAsigurat=?, dataInceput=?, dataSfarsit=?, seriePolita=?, sumaAsigurata=?, "
		            "primaAsig=? WHERE numarPolita=?;",
		            (nume_asigurat, prenume_asigurat, valabilitate_inceput, valabilitate_sfarsit, serie_polita, suma_asig, prima_asig, nr_polita))
		conn.commit()

def view_polite():
	with conn.cursor() as cur:
		cur.execute("SELECT numeAsigurat, prenumeAsigurat, seriePolita, numarPolita, sumaAsigurata, "
		            "primaAsig , dataInceput, dataSfarsit, idUser FROM Polite ;")
		data=cur.fetchall()
		return data
		"""for row in cur:
			row_to_list = [elem for elem in row]
		return row_to_list"""

def view_serie_nr_polite(idUser):
	with conn.cursor() as cur:
		cur.execute("SELECT seriePolita, numarPolita FROM Polite WHERE idUser=?;", (idUser))
		data=cur.fetchall()
		return data

def get_polita_by_nr(seriePolita,numarPolita):
	with conn.cursor() as cur:
		cur.execute("SELECT numeAsigurat, prenumeAsigurat, seriePolita, numarPolita, sumaAsigurata, "
		            "primaAsig , dataInceput, dataSfarsit  FROM Polite WHERE seriePolita=? AND numarPolita=?;", (seriePolita,numarPolita))
		data=cur.fetchall()
		return data

def delete_polita(seriePolita,numarPolita):
	with conn.cursor() as cur:
		cur.execute("DELETE FROM Polite WHERE seriePolita=? AND numarPolita=?;", (seriePolita,numarPolita))
		conn.commit()