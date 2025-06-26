# importation de package
import pandas as pd
import numpy as np
import openpyxl as xl
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
# upload data
df=pd.read_excel('C:\\Users\\kensc\\Downloads\\COURSERA\\PYTHON\ProjetArima\\RevenueDataDAS.xlsx',engine='openpyxl')
print(df)
df.info ()
df.isnull().sum()
df.set_index('Mois',inplace=True)

#afficher un graphique
plt.figure(figsize=(10,6))
plt.plot(df.index,df['Recettes'], color='blue')
plt.title('Evoltion des recettes mensuelles de 2005 a 2024')
plt.xlabel('Mois')
plt.ylabel("Recettes")
plt.grid(True)
plt.show()


#result = adfuller(series)
#statistic, pval, lags, nobs, crit_vals, aic = result

#test de dickey fuller augmenté
print("dckey fuller sur serie originale")
testDick = adfuller(df['Recettes'])

table=[
    ['Valeur de test',testDick[0]],
    ['P-Value',testDick[1]],
    ['Nb de decalage utilise',testDick[2]],
    ['Nb observations',testDick[3]],
    ['critical values',testDick[4]],
    ['IC best',testDick[5]],
    ['Conclusion','La serie est stationnaire'if testDick[1]<0.05 else 'La serie est non stationnaire']
]
# afficher resultat test format tableau

print(tabulate(table,headers=['Metrique','Valeur'],tablefmt='tsv'))

#effectuer la decomposition
decomposition=seasonal_decompose(df["Recettes"],model='additive')

#extraire les composantes de la decomposition
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Recettes'],label='Serie originale a')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label="Tendance")
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label="Saisonnalite")
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label="Residus")
plt.legend(loc='best')

plt.tight_layout()
plt.show()

#effectuer la decomposition multiplicative
decomposition=seasonal_decompose(df["Recettes"],model='multiplicative')

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Recettes'],label='Serie originale m')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label="Tendance")
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label="Saisonnalite")
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label="Residus")
plt.legend(loc='best')

plt.tight_layout()
plt.show()

#creer les subplots

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))

#tracer l'ACF

plot_acf(df['Recettes'],lags=30,zero=True, ax=ax1)
ax1.set_title('ACF - Recettes')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Correlation')
ax1.grid(True)
ax1.set_xticks(np.arange(0,31,1))

#tracer le PACF
plot_pacf(df['Recettes'],lags=30,zero=True, ax=ax2)
ax2.set_title('PACF - Recettes')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Correlation partielle')
ax2.grid(True)
ax2.set_xticks(np.arange(0,31,1))

plt.tight_layout()
plt.show()

# pasage au log
logdf= np.log(df['Recettes'])

#afficher la serie log

plt.figure(figsize=(10,6))
plt.plot(logdf)
plt.title('Serie log')
plt.xlabel('Date')
plt.ylabel('Recettes')
plt.grid(True)
plt.show()


 #test de dickey fuller augmenté
print("Dickey fuller au passage au log")
testDick1 = adfuller(logdf)

table1=[
    ['Valeur de test',testDick1[0]],
    ['P-Value',testDick1[1]],
    ['Nb de decalage utilise',testDick1[2]],
    ['Nb observations',testDick1[3]],
    ['critical values',testDick1[4]],
    ['IC best',testDick1[5]],
    ['Conclusion','La serie est stationnaire'if testDick1[1] < 0.05 else 'La serie est non stationnaire']
]
# afficher resultat test format tableau

print(tabulate(table1,headers=['Metrique','Valeur'],tablefmt='tsv'))  


# difference premiere du log
diff1= logdf.diff().dropna()

#afficher la serie differenciee

plt.figure(figsize=(10,6))
plt.plot(diff1)
plt.title('Serie diffenciee(log)')
plt.xlabel('Date')
plt.ylabel('Recette')
plt.grid(True)
plt.show()


 #test de dickey fuller augmenté
print("Dickey fuller en difference premiere de la serie log")
testDick2 = adfuller(diff1)

table2=[
    ['Valeur de test',testDick2[0]],
    ['P-Value',testDick2[1]],
    ['Nb de decalage utilise',testDick2[2]],
    ['Nb observations',testDick2[3]],
    ['critical values',testDick2[4]],
    ['IC best',testDick2[5]],
    ['Conclusion','La serie est stationnaire'if testDick2[1] < 0.05 else 'La serie est non stationnaire']
]
# afficher resultat test format tableau

print(tabulate(table2,headers=['Metrique','Valeur'],tablefmt='tsv'))          

# differenciation premiere(sans log)
differenced1= df['Recettes'].diff().dropna()

#afficher la serie differenciee

plt.figure(figsize=(10,6))
plt.plot(differenced1)
plt.title('Serie diffenciee (sans log)')
plt.xlabel('Date')
plt.ylabel('Recettes')
plt.grid(True)
plt.show()


 #test de dickey fuller augmenté
print("Dickey fuller en differenciation premiere(sans log)")
testDick3 = adfuller(differenced1)

table3=[
    ['Valeur de test',testDick3[0]],
    ['P-Value',testDick3[1]],
    ['Nb de decalage utilise',testDick3[2]],
    ['Nb observations',testDick3[3]],
    ['critical values',testDick3[4]],
    ['IC best',testDick3[5]],
    ['Conclusion','La serie est stationnaire'if testDick3[1] < 0.05 else 'La serie est non stationnaire']
]
# afficher resultat test format tableau

print(tabulate(table3,headers=['Metrique','Valeur'],tablefmt='tsv'))



#creer les subplots pour la differenciations sans log

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,8))

#tracer l'ACF

plot_acf(differenced1,lags=50,zero=True, ax=ax1)
ax1.set_title('ACF - Recettes')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Correlation en diff 1')
ax1.grid(True)
ax1.set_xticks(np.arange(0,51,1))

#tracer le PACF
plot_pacf(differenced1,lags=50,zero=True, ax=ax2)
ax2.set_title('PACF - Recettes')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Correlation partielle en diff 1')
ax2.grid(True)
ax2.set_xticks(np.arange(0,51,1))

plt.tight_layout()
plt.show()

#separer les donnees en ensemble d'apprentissage et ensemble de test
train_data=df['Recettes'][:-12]
test_data= df['Recettes'][-12:]

print("modele test")
#estimation du modele
model= ARIMA(train_data,order=(2,1,1))
model_fit = model.fit()
print(model_fit.summary())


#separer les donnees en ensemble d'apprentissage et ensemble de test
train_data1=df['Recettes'][:-12]
test_data1= df['Recettes'][-12:]

print("modele automatique")
# modele automatique
model1=pm.auto_arima(train_data1)
print(model1.summary())

#ACF Pour les residus
#ajuster le modele
model1.fit(train_data1)
#calculer les residus
residual= model1.resid()
#tracer la fonction ACF des residus
plot_acf(residual,lags=20)
plt.xlabel('Lag')
plt.ylabel('Autocorelation')
plt.title('ACF des residus')
plt.show()

#tracer la fonction PACF des residus
plot_pacf(residual,lags=20)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorelation')
plt.title('PACF des residus')
plt.show()


#Faire des predictions sur l'ensemble d'entrainement
train_pred,train_confint = model1.predict_in_sample(return_conf_int=True)
#Faire des predictions sur l'esemble test
n_period=len(test_data1)
predicted,confint= model1.predict(n_period=n_period,return_conf_int=True)
#Contatener les prediction de l'ensemble d'entrainement et de test
all_predictions=pd.concat([pd.Series(train_pred,index=train_data1.index),pd.Series(predicted,index=test_data1.index)],axis=0)

#tracer les valeurs reelles et les predictions pour l'ensemble d'entrainement et de test

plt.figure(figsize=(12,6))
plt.plot(train_data1,label='Oberved Training',color='blue')
plt.plot(test_data1,label='Observed Test',color='green')
plt.plot(all_predictions,label='Predicted',color='red')

plt.xlabel('Mois')
plt.ylabel('Recettes')
plt.title('Observed VS Predicted Recettes')
plt.legend()
plt.grid(True)
plt.show()

#separer les donnees en ensemble d'apprentissage et ensemble de test
train_data2=df['Recettes'][:-12]
test_data2= df['Recettes'][-12:]

print("modele SARIMA")
#estimation du modele
model2= SARIMAX(train_data2,order=(0,1,2),seasonal_order=(0,0,0,12))
model_fit2 = model2.fit()
print(model_fit2.summary())

#ACF Pour les residus
#ajuster le modele
model2.fit(train_data2)
#calculer les residus
residual1= model_fit2.resid
#tracer la fonction ACF des residus
plot_acf(residual1,lags=20)
plt.xlabel('Lag')
plt.ylabel('Autocorelation')
plt.title('ACF des residus')
plt.show()

#tracer la fonction PACF des residus
plot_pacf(residual1,lags=20)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorelation')
plt.title('PACF des residus')
plt.show()


#Faire des predictions sur l'ensemble d'entrainement
train_pred = model_fit2.fittedvalues
#Faire des predictions sur l'esemble test
n_period = len(test_data2)
predicted = model_fit2.forecast(steps=n_period)
#Contatener les prediction de l'ensemble d'entrainement et de test
all_predictions=pd.concat([pd.Series(train_pred,index=train_data2.index),pd.Series(predicted,index=test_data2.index)],axis=0)

#tracer les valeurs reelles et les predictions pour l'ensemble d'entrainement et de test

plt.figure(figsize=(12,6))
plt.plot(train_data2,label='Oberved Training',color='blue')
plt.plot(test_data2,label='Observed Test',color='green')
plt.plot(all_predictions,label='Predicted',color='red')

plt.xlabel('Mois')
plt.ylabel('Recettes')
plt.title('Observed VS Predicted Recettes')
plt.legend()
plt.grid(True)
plt.show()