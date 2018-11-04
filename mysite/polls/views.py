from django.shortcuts import render
from django.http import HttpResponse
# Import some common packages
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import urllib

# Import ML packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pandas.plotting import scatter_matrix

#{% load staticfiles %}

from .future_encoders import ColumnTransformer
from .future_encoders import OneHotEncoder

def index(request):
	response = getHead()
	#print("test")
	return HttpResponse("<b>Fitbit biometric data analysis:</b></br>" + str(response))

def getHead():
	result = " </br>"
	
	fitbitData = pd.read_csv("~/Documents/healthhacks/mysite/polls/MasterDatabase.csv")
	fitbitData = fitbitData.drop(['Unnamed: 0'], axis=1)
	

	fitbitData_new = fitbitData.drop(['Calories', 'Steps', 'Distance', 'Minuteslightlyactive', 'Minutesfairlyactive', 'Minutesveryactive'], axis=1)
	fitbitData_num = fitbitData_new.drop(['Date', 'Week Day'], axis=1)

	num_attribs = list(fitbitData_num)
	cat_attribs = ['Date','Week Day']

	num_pipeline = Pipeline([
        ('imputer', Imputer(strategy = "median")),
        ('std_scaler', StandardScaler()),
	])

	num_pipeline.fit_transform(fitbitData_num)

	full_pipeline = ColumnTransformer([
    	("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
	])

	fitbitData_prepared = full_pipeline.fit_transform(fitbitData_new)

	label = "Minutessedentary" 

	train_set, test_set = train_test_split(fitbitData, test_size=0.2, random_state=10)

	x_tr = train_set.drop(label,axis=1)
	y_train = train_set[label].copy()
	x_te = test_set.drop(label,axis=1)
	y_test = test_set[label].copy()

	x_train = full_pipeline.transform(x_tr) #Process training data
	x_test = full_pipeline.transform(x_te) #Process test data
	
	from sklearn.ensemble import RandomForestRegressor
	import statistics
	#Create a Random Forest Regression based on the training data
	#print("RandomForest Regressor")
	forest_reg = RandomForestRegressor(random_state=42)
	forest_reg.fit(x_train,y_train)

	predictedSit = round(statistics.mean(forest_reg.predict(x_test)),2)
	result += "Predicted sedentary level today is " + str(predictedSit) + " minutes.</br>"
	
	if predictedSit > 960:
		activityLevel = 1.2
	else:
		activityLevel = 1.55 #Assumed for sake of simplicity
	
	#BMRcal * AL = calories to maintain weight
	neutralCalories = 1459 * activityLevel

	fitbitData_num_ = fitbitData.drop(['Steps', 'Distance', 'Minutessedentary', 'Minuteslightlyactive', 'Minutesfairlyactive', 'Minutesveryactive','Date', 'Week Day'], axis = 1)

	num_attribs = list(fitbitData_num)
	cat_attribs = ['Date','Week Day']

	num_pipeline = Pipeline([
        ('imputer', Imputer(strategy = "median")),
        ('std_scaler', StandardScaler()),
	])

	full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
	])

	fitbitData_prepared = full_pipeline.fit_transform(fitbitData)

	label = "Calories" 

	train_set, test_set = train_test_split(fitbitData, test_size=0.2, random_state=10)

	x_tr = train_set.drop(label,axis=1)
	y_train = train_set[label].copy()
	x_te = test_set.drop(label,axis=1)
	y_test = test_set[label].copy()

	x_train = full_pipeline.transform(x_tr) #Process training data
	x_test = full_pipeline.transform(x_te) #Process test data

	attributes = num_attribs


	#Create a Random Forest Regression based on the training data
	#print("RandomForest Regressor")
	forest_reg = RandomForestRegressor(random_state=42)
	forest_reg.fit(x_train,y_train)

	predictedCalories = round(statistics.mean(forest_reg.predict(x_test)),2)
	result += "Predicted calories burned for today " + str(predictedCalories) + " calories.</br>"

	result += "For this individual to maintain their weight, they must consume " + str(round(neutralCalories,2)) + " calories.</br>"

	dif = predictedCalories - neutralCalories
	if dif > 100:
		result += "This individual is predicted to have an active day, {recommendation goes here}</br>"
	elif dif < -100:
		result += "This individual is predicted to have an inactive day, {recommendation goes here}</br>"
	else:
		result += "No changes recommended."
	return(result)
'''
	if caloriesPredicted > someNumber:
		result += "An excess of calories burned is predicted, more calories rich food could be eaten to prevent weight loss.</br> "
	elif caloriesPredicted < someNumber:
		result += "Not enough calories being burned is predicted, lighter food could help prevent unwanted weight gain.</br> "
	else:
		result += "A good amount of calories being burned is predicted, no change recommended."


	if hoursNotMoving > anotherNumber:
		result += "Hours predicted to be sedentary are high, recommend lower calorie and more filling food."
	elif hoursNotMoving < anotherNumber:
		result += "Hours predicted to sedentary are low, recommend higher calorie food."
	else:
		result += "A good amount of sedentary hours are predicted, no change recommended."
'''

	#return(result)
# Create your views here.
