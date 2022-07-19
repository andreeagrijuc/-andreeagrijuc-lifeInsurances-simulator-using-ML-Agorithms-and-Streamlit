import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score , precision_score, recall_score, classification_report, confusion_matrix,r2_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from lime import lime_tabular
import pickle
import time
import joblib

#region Dataset

dataset = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\dataset_analiza.csv')
dataset['Modified_Response'] = dataset['Response'].apply(
		lambda x: 0 if x <=6  and x >= 0 else (1 if x == 8 or x==7 else -1))
#print(dataset['Modified_Response'].value_counts())
dataset.drop(['Response'],axis = 1, inplace=True)
#dataset.drop(['Id'],axis = 1, inplace=True)
missing_val_count_by_column = dataset.isnull().sum()/len(dataset)
#print("===========================")
#print(missing_val_count_by_column)
#print("===========================")
#print(missing_val_count_by_column[missing_val_count_by_column > 0.4].sort_values(ascending=False))
dataset2=dataset
X = dataset.drop(labels='Modified_Response',axis=1)
Y = dataset['Modified_Response']
X = X.fillna(X.mean())
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=1)

#endregion

#region Dataset2
#dataset2.drop(['Id'],axis=1,inplace=True)
#print(dataset2.head(15))
test_data=dataset2.head(20000)
#print(test_data)
A = test_data.drop(labels='Modified_Response',axis=1)
B = test_data['Modified_Response']
A = A.fillna(A.mean())
A_train, A_test, B_train, B_test = train_test_split(A,B,test_size = 0.25, random_state=1)

#endregion

#region Reduced Dataset
dataset_for_input=pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\train_red_cols.csv')
dataset_for_input.drop(['Id'],axis=1,inplace=True)

dataset_for_result = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\input_dataset.csv')
dataset_for_result.drop(['Id'],axis=1,inplace=True)

dataset_for_input['Modified_Response'] = dataset_for_input['Response'].apply(
		lambda x: 0 if x <=6  and x >= 0 else (1 if x == 8 or x==7 else -1))
dataset_for_input.drop('Response', axis = 1, inplace=True)
missing_val_count_by_column2 = dataset_for_input.isnull().sum() / len(dataset_for_input)
train_cols = dataset_for_input.drop(labels='Modified_Response', axis=1)
train_cols = train_cols.fillna(train_cols.mean())

train_cols_for_SVM=train_cols.head(5000)
train_resp = dataset_for_input['Modified_Response']
train_resp_for_SVM=train_resp.head(5000)
test_dt = dataset_for_result.drop(labels='Modified_Response', axis=1)
test_result = dataset_for_result['Modified_Response']
#check=test_result.isnull().sum().sum()
#print(train_cols_for_SVM)
#endregion


def check_scores(model, X_train, X_test):
	# Making predictions on train and test data

	train_class_preds = model.predict(X_train)
	test_class_preds = model.predict(X_test)

	# Get the probabilities on train and test
	train_preds = model.predict_proba(X_train)[:, 1]
	test_preds = model.predict_proba(X_test)[:, 1]

	# Calculating accuracy on train and test
	train_accuracy = accuracy_score(Y_train, train_class_preds)
	test_accuracy = accuracy_score(Y_test, test_class_preds)

	print("The accuracy on train dataset is", train_accuracy)
	print("The accuracy on test dataset is", test_accuracy)
	print()
	# Get the confusion matrices for train and test
	train_cm = confusion_matrix(Y_train, train_class_preds)
	test_cm = confusion_matrix(Y_test, test_class_preds)

	#print('Train confusion matrix:')
	#print(train_cm)
	print()
	print('Test confusion matrix:')
	print(test_cm)
	print()
	print('FP', test_cm[0][1])
	print('TN', test_cm[1][1])
	specificity = test_cm[1][1]/(test_cm[1][1]+test_cm[0][1])

	# Get the roc_auc score for train and test dataset
	train_auc = roc_auc_score(Y_train, train_preds)
	test_auc = roc_auc_score(Y_test, test_preds)

	print('ROC on train data:', train_auc)
	print('ROC on test data:', test_auc)

	# Fscore, precision and recall on test data
	f1 = f1_score(Y_test, test_class_preds)
	precision = precision_score(Y_test, test_class_preds)
	recall = recall_score(Y_test, test_class_preds)

	# R2 score on train and test data
	train_log = log_loss(Y_train, train_preds)
	test_log = log_loss(Y_test, test_preds)

	print()
	print('Train log loss:', train_log)
	print('Test log loss:', test_log)
	print()
	print("F score is:", f1)
	print("Precision is:", precision)
	print("Recall is:", recall)
	print('specificity', specificity)
	return model, train_auc, test_auc, train_accuracy, test_accuracy, f1, precision, recall, train_log, test_log,specificity

def check_scores3(filename, train_cols, test_dt):
	#loaded_model = pickle.load(open(filename, 'rb'))

	# Making predictions on train and test data
	train_class_preds = filename.predict(train_cols)
	test_class_preds = filename.predict(test_dt)
	test_class_preds=np.array(test_class_preds)

	#print('test_class_preds',test_class_preds)

	# Get the probabilities on train and test
	#train_preds = loaded_model.predict_proba(train_cols)[:, 1]
	#test_preds = loaded_model.predict_proba(test_dt)[:, 1]
	# Calculating accuracy on train and test
	#train_accuracy = accuracy_score(train_resp, train_class_preds)
	#test_accuracy = accuracy_score(test_result, test_class_preds)

	return test_class_preds[0]
def check_scores_for_SVM(model, train_cols_for_SVM, test_dt):

	# Making predictions on train and test data
	train_class_preds = model.predict(train_cols_for_SVM)
	test_class_preds = model.predict(test_dt)
	test_class_preds=np.array(test_class_preds)

	#print('test_class_preds',test_class_preds)

	# Get the probabilities on train and test
	train_preds = model.predict_proba(train_cols_for_SVM)[:, 1]
	test_preds = model.predict_proba(test_dt)[:, 1]

	print('test_preds', test_preds)

	return test_class_preds[0]

def check_importance(model, X_train):
	# Checking importance of features
	importances = model.feature_importances_

	# List of columns and their importances
	importance_dict = {'Feature': list(X_train.columns),
	                   'Feature Importance': importances}
	# Creating a dataframe
	dataset = pd.DataFrame(importance_dict)

	# Rounding it off to 2 digits as we might get exponential numbers
	dataset['Feature Importance'] = round(dataset['Feature Importance'], 2)
	return dataset.sort_values(by=['Feature Importance'], ascending=False)
def check_importance3(model, train_cols):
	# Checking importance of features
	importances = model.feature_importances_

	# List of columns and their importances
	importance_dict = {'Feature': list(train_cols.columns),
	                   'Feature Importance': importances}
	# Creating a dataframe
	dataset_for_input = pd.DataFrame(importance_dict)

	# Rounding it off to 2 digits as we might get exponential numbers
	dataset_for_input['Feature Importance'] = round(dataset_for_input['Feature Importance'], 2)
	return dataset_for_input.sort_values(by=['Feature Importance'], ascending=False)

def grid_search(model, parameters, X_train, Y_train):
	time_start1 = time.time()
	grid = GridSearchCV(estimator=model,
	                    param_grid=parameters,
	                    cv=2, verbose=2, scoring='roc_auc')
	grid.fit(X_train, Y_train)
	optimal_model = grid.best_estimator_
	print('Best parameters are: ')
	print(grid.best_params_)
	time_start2 = time.time()
	duration=time_start2-time_start1
	print('duration: ', duration)
	return optimal_model, duration

def grid_search_LR(model, parameters, train_cols, train_resp):
	# Doing a grid
	grid = GridSearchCV(estimator=model,
	                    param_grid=parameters,
	                    cv=2, verbose=2, scoring='roc_auc')
	# Fitting the grid
	grid.fit(train_cols, train_resp)

	# Best model found using grid search
	optimal_model = grid.best_estimator_
	#print('Best parameters are: ')
	#print(grid.best_params_)

	#filename = 'finalized_model_LR.sav'
	#pickle.dump(optimal_model, open(filename, 'wb'))
	filename = joblib.dump(optimal_model, 'finalized_model_LR.pkl')
	return filename
def grid_search_RF(model, parameters, train_cols, train_resp):
	# Doing a grid
	grid = GridSearchCV(estimator=model,
	                    param_grid=parameters,
	                    cv=2, verbose=2, scoring='roc_auc')
	# Fitting the grid
	grid.fit(train_cols, train_resp)

	# Best model found using grid search
	optimal_model = grid.best_estimator_
	#print('Best parameters are: ')
	#print(grid.best_params_)

	#filename = 'finalized_model_LR.sav'
	#pickle.dump(optimal_model, open(filename, 'wb'))
	filename = joblib.dump(optimal_model, 'finalized_model_RF.pkl')
	return filename
def grid_search_SVM(model, parameters, train_cols, train_resp):
	# Doing a grid
	grid = GridSearchCV(estimator=model,
	                    param_grid=parameters,
	                    cv=2, verbose=2, scoring='roc_auc')
	# Fitting the grid
	grid.fit(train_cols, train_resp)

	# Best model found using grid search
	optimal_model = grid.best_estimator_
	#print('Best parameters are: ')
	#print(grid.best_params_)

	#filename = 'finalized_model_LR.sav'
	#pickle.dump(optimal_model, open(filename, 'wb'))
	filename = joblib.dump(optimal_model, 'finalized_model_SVM.pkl')
	return filename
def grid_search3(model, parameters, train_cols, train_resp):
	# Doing a grid
	grid = GridSearchCV(estimator=model,
	                    param_grid=parameters,
	                    cv=2, verbose=2, scoring='roc_auc')
	# Fitting the grid
	grid.fit(train_cols, train_resp)
	# Best model found using grid search
	optimal_model = grid.best_estimator_
	#print('Best parameters are: ')
	#print(grid.best_params_)
	#filename = 'finalized_model_LR.sav'
	#pickle.dump(optimal_model, open(filename, 'wb'))
	return optimal_model


#region Logistic Regression
def LRegr():
	solvers = ['lbfgs']
	penalty = ['l2']
	c_values = [1.0]
	lr_parameters = dict(solver=solvers,penalty=penalty,C=c_values)

	lr_optimal_model, lr_duration = grid_search(LogisticRegression( max_iter=5000), lr_parameters, X_train, Y_train)

	lr_model, lr_train_auc, lr_test_auc, lr_train_accuracy, lr_test_accuracy,lr_f1, lr_precision, lr_recall,lr_train_log, \
		lr_test_log, lr_specificity = check_scores(lr_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	lr_list = {
		'Id': 2,
		'Model Name': 'Logistic Regression',
		'Train ROC': lr_train_auc,
		'Test ROC': lr_test_auc,
	    'Train Accuracy': lr_train_accuracy,
		'Test Accuracy' : lr_test_accuracy,
		'Train Log Loss' : lr_train_log,
		'Test Log Loss' : lr_test_log,
	    'F-Score': lr_f1,
	    'Precision': lr_precision,
		'Recall': lr_recall,
		'Specificity': lr_specificity,
		'Duration': lr_duration }

	#prediction = pd.DataFrame(lr_list, columns=['Id','Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall']).to_csv('prediction.csv')
	analiza_expl = analiza_expl.append(lr_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#LRegr()
#endregion
#region Logistic Regression2

def LRegr2():
	solvers = ['lbfgs']
	penalty = ['l2']
	c_values = [1.0]
	lr_parameters = dict(solver=solvers,penalty=penalty,C=c_values)

	#lr_optimal_model = grid_search_LR(LogisticRegression( max_iter=10000), lr_parameters, train_cols, train_resp)
	loaded_model = joblib.load('finalized_model_LR.pkl')
	test_class_preds = check_scores3(loaded_model, train_cols, test_dt)

	print(test_class_preds)

	return test_class_preds


#endregion

#region Random Forest
def RandForest():
	# Number of trees
	n_estimators = [50]

	# Maximum depth of trees
	max_depth = [10]

	# Minimum number of samples required to split a node
	min_samples_split = [100]

	# Minimum number of samples required at each leaf node
	min_samples_leaf = [40]

	# Hyperparameter Grid
	rf_parameters = {'n_estimators' : n_estimators,
	              'max_depth' : max_depth,
	              'min_samples_split' : min_samples_split,
	              'min_samples_leaf' : min_samples_leaf}
	#print(rf_parameters)
	#finding the best model
	rf_optimal_model, rf_duration = grid_search(RandomForestClassifier(), rf_parameters, X_train, Y_train)
	# Getting scores from all the metrices
	rf_model, rf_train_auc, rf_test_auc, rf_train_accuracy, rf_test_accuracy,rf_f1, rf_precision,rf_recall,rf_train_log, \
		rf_test_log, rf_specificity = check_scores(rf_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	rf_list = {
		'Id': 2,
		'Model Name': 'Radom Forests',
		'Train ROC': rf_train_auc,
		'Test ROC': rf_test_auc,
		'Train Accuracy': rf_train_accuracy,
		'Test Accuracy': rf_test_accuracy,
		'Train Log Loss': rf_train_log,
		'Test Log Loss': rf_test_log,
		'F-Score': rf_f1,
		'Precision': rf_precision,
		'Recall': rf_recall,
		'Specificity': rf_specificity,
		'Duration': rf_duration}

	# prediction = pd.DataFrame(lr_list, columns=['Id','Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall']).to_csv('prediction.csv')
	analiza_expl = analiza_expl.append(rf_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#RandForest()
#endregion
#region Random Forest2
def RandForest2():
	# Number of trees
	n_estimators = [80]
	# Maximum depth of trees
	max_depth = [8]
	# Minimum number of samples required to split a node
	min_samples_split = [100]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [40]
	# Hyperparameter Grid
	rf_parameters = {'n_estimators' : n_estimators,
	              'max_depth' : max_depth,
	              'min_samples_split' : min_samples_split,
	              'min_samples_leaf' : min_samples_leaf}
	#print(rf_parameters)
	#finding the best model
	#rf_optimal_model = grid_search_RF(RandomForestClassifier(), rf_parameters, train_cols, train_resp)
	# Getting scores from all the metrices
	loaded_model = joblib.load('finalized_model_RF.pkl')
	test_class_preds = check_scores3(loaded_model, train_cols, test_dt )
	print(test_class_preds)
	#file_read = f'C:\\Users\\andre\\PycharmProjects\\SmartInsurance\\finalized_model_RF.pkl'
	#objects = pd.read_pickle(file_read)
	#print(objects)
	return test_class_preds
RandForest2()
#endregion

#region Decision Tree
def DecTree():
	# Maximum depth of trees
	max_depth = [10]
	# Minimum number of samples required to split a node
	min_samples_split = [250]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [30]
	# Hyperparameter Grid
	dt_parameters = {'max_depth' : max_depth,
	              'min_samples_split' : min_samples_split,
	              'min_samples_leaf' : min_samples_leaf}
	#print(dt_parameters)
	#finding the best model
	dt_optimal_model, dt_duration = grid_search(DecisionTreeClassifier(), dt_parameters, X_train, Y_train)
	dt_model, dt_train_auc, dt_test_auc, dt_train_accuracy, dt_test_accuracy,dt_f1, dt_precision,dt_recall,dt_train_log, \
		dt_test_log, dt_specificity = check_scores(dt_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	dt_list = {
		'Id': 4,
		'Model Name': 'Decision Trees',
		'Train ROC': dt_train_auc,
		'Test ROC': dt_test_auc,
		'Train Accuracy': dt_train_accuracy,
		'Test Accuracy': dt_test_accuracy,
		'Train Log Loss': dt_train_log,
		'Test Log Loss': dt_test_log,
		'F-Score': dt_f1,
		'Precision': dt_precision,
		'Recall': dt_recall,
		'Specificity': dt_specificity,
		'Duration': dt_duration}

	analiza_expl = analiza_expl.append(dt_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#DecTree()
#endregion

#region kNN
def kNN():
	n_neighbors=[35]
	leaf_size=[15]
	kNN_parameters={'n_neighbors': n_neighbors,
	                'leaf_size': leaf_size}
	kNN_optimal_model, knn_duration=grid_search(KNeighborsClassifier(), kNN_parameters, X_train, Y_train)
	knn_model, knn_train_auc, knn_test_auc,knn_train_accuracy, knn_test_accuracy,knn_f1, knn_precision,knn_recall,knn_train_log, \
		knn_test_log, knn_specificity = check_scores(kNN_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	knn_list = {
		'Id': 4,
		'Model Name': 'kNN',
		'Train ROC': knn_train_auc,
		'Test ROC': knn_test_auc,
		'Train Accuracy': knn_train_accuracy,
		'Test Accuracy': knn_test_accuracy,
		'Train Log Loss': knn_train_log,
		'Test Log Loss': knn_test_log,
		'F-Score': knn_f1,
		'Precision': knn_precision,
		'Recall': knn_recall,
		'Specificity': knn_specificity,
		'Duration': knn_duration}

	# prediction = pd.DataFrame(lr_list, columns=['Id','Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall']).to_csv('prediction.csv')
	analiza_expl = analiza_expl.append(knn_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#kNN()
#endregion

#region MLP Classifier
def MLP():
	solver=['adam']
	activation = ['relu']
	hidden_layer_sizes=[150]
	max_iter=[5000]
	random_state=[1]
	mlp_parameters = {
		'solver':solver,
		'activation': activation,
		'hidden_layer_sizes':hidden_layer_sizes,
		'random_state':random_state,
		'max_iter': max_iter
	}
	mlp_optimal_model, mlp_duration=grid_search(MLPClassifier(),mlp_parameters,X_train,Y_train)
	mlp_model, mlp_train_auc, mlp_test_auc, mlp_train_accuracy, mlp_test_accuracy,mlp_f1, mlp_precision,mlp_recall,mlp_train_log, \
		mlp_test_log, mlp_specificity = check_scores(mlp_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	mlp_list = {
		'Id': 4,
		'Model Name': 'MLP',
		'Train ROC': mlp_train_auc,
		'Test ROC': mlp_test_auc,
		'Train Accuracy': mlp_train_accuracy,
		'Test Accuracy': mlp_test_accuracy,
		'Train Log Loss': mlp_train_log,
		'Test Log Loss': mlp_test_log,
		'F-Score': mlp_f1,
		'Precision': mlp_precision,
		'Recall': mlp_recall,
		'Specificity': mlp_specificity,
		'Duration': mlp_duration}

	# prediction = pd.DataFrame(lr_list, columns=['Id','Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall']).to_csv('prediction.csv')
	analiza_expl = analiza_expl.append(mlp_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#MLP()
#endregion

#region SVM
def SVM():
	kernel=['rbf']
	gamma=[1.0]
	probability = [True]
	C=[1.0]
	svm_parameters = {
		'kernel': kernel,
		'gamma': gamma,
		'probability': probability,
		'C': C
	}
	svm_optimal_model, svm_duration=grid_search(svm.SVC(),svm_parameters,X_train,Y_train)
	#print(svm_optimal_model)
	svm_model, svm_train_auc, svm_test_auc, svm_train_accuracy, svm_test_accuracy, svm_f1, svm_precision,svm_recall,svm_train_log, \
		svm_test_log,svm_specificity = check_scores(svm_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	svm_list = {
		'Id': 8,
		'Model Name': 'SVM',
		'Train ROC': svm_train_auc,
		'Test ROC': svm_test_auc,
		'Train Accuracy': svm_train_accuracy,
		'Test Accuracy': svm_test_accuracy,
		'Train Log Loss': svm_train_log,
		'Test Log Loss': svm_test_log,
		'F-Score': svm_f1,
		'Precision': svm_precision,
		'Recall': svm_recall,
		'Specificity': svm_specificity,
		'Duration': svm_duration}

	# prediction = pd.DataFrame(lr_list, columns=['Id','Model Name', 'Train ROC', 'Test ROC', 'Train Accuracy', 'Test Accuracy', 'Train Log Loss','Test Log Loss','F-Score', 'Precision','Recall']).to_csv('prediction.csv')
	analiza_expl = analiza_expl.append(svm_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
	#clf = svm.SVC(kernel='linear')
	#clf.fit(A_train, B_train)
	#Y_pred = clf.predict(A_test)
	#print(Y_pred)
#SVM()
#endregion
#region SVM2
def SVM2():
	kernel=['rbf']
	gamma=[1.0]
	probability = [True]
	C=[1.0]
	svm_parameters = {
		'kernel': kernel,
		'gamma': gamma,
		'probability': probability,
		'C': C
	}
	loaded_model = joblib.load('finalized_model_SVM.pkl')
	#svm_optimal_model=grid_search_SVM(svm.SVC(),svm_parameters,train_cols_for_SVM,train_resp_for_SVM)
	test_class_preds = check_scores_for_SVM(loaded_model, train_cols_for_SVM, test_dt )
	return test_class_preds

#SVM2()
#endregion

#region NBayes
def NBayes():
	gnb = GaussianNB()
	sample_weigh = [0.4]
	#y_pred = gnb.fit(X_train, Y_train).predict(X_test)
	#print("Number of mislabeled points out of a total %d points : %d" (X_test.shape[0], (Y_test != y_pred).sum()))
	nb_parameters = {}
	nb_optimal_model, nb_duration=grid_search(CategoricalNB(),nb_parameters,X_train,Y_train)
	nb_model, nb_train_auc, nb_test_auc, nb_train_accuracy, nb_test_accuracy,nb_f1, nb_precision,nb_recall,nb_train_log, \
		nb_test_log, nb_specificity = check_scores(nb_optimal_model, X_train, X_test )
	analiza_expl_file = f'C:\\Users\\andre\\Desktop\\licenta\\prediction.csv'
	analiza_expl = pd.read_csv('C:\\Users\\andre\\Desktop\\licenta\\prediction.csv')
	nb_list = {
		'Id': 1,
		'Denumire Model': 'CategoricalNB',
		'Train Accuracy': nb_train_accuracy,
		'Test Accuracy': nb_test_accuracy,
		'F-Score': nb_f1,
		'Precision': nb_precision,
		'Recall': nb_recall,
		'Specificity': nb_specificity,
		'Duration': nb_duration}


	analiza_expl = analiza_expl.append(nb_list, ignore_index=True)
	analiza_expl.to_csv(analiza_expl_file, index=False)
#NBayes()
#endregion