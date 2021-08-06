# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:40:51 2021

@author: amart
"""
from sklearn.metrics import mean_squared_error
import seaborn as sb
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from math import sqrt

dat=pd.read_csv("F:\Data Science Assignments\Python-Assignment\Multi-Linear Regression\Startups.csv")
new=dat.loc[:,dat.columns !='State']
new=new.rename(columns={'R&D Spend':'rnd','Administration':'admin','Marketing Spend':'market'})
sb.pairplot(new)
new.corr()


model1=smf.ols('Profit~rnd+admin+market',data=new).fit()
model1.summary()                  #Rsquare =0.951   and  Adj Rsquare =0.948
pred1=model1.predict(new)
pred1.corr(new.Profit)            #0.9750620462659412
rmse=sqrt(mean_squared_error(new.Profit, pred1))
rmse                              #8855.344489015142

Ad_model=smf.ols('Profit~admin',data=new).fit()
Ad_model.summary()

sm.graphics.influence_plot(model1) #Finding the outliers
new=new.drop(new.index[[45,49]],axis=0) #Removing Outliers

#Calculating VIF Scores
rsq_admin=smf.ols('admin~rnd+market',data=new).fit().rsquared
vif_admin=(1/(1-rsq_admin))        #1.1768900319565039
rsq_market=smf.ols('market~rnd+admin',data=new).fit().rsquared
vif_market=(1/(1-rsq_market))      #2.117398
rsq_rnd=smf.ols('rnd~market+admin',data=new).fit().rsquared
vif_rnd=(1/(1-rsq_rnd))            #2.260983
ds={'Index':['Admin','Market','R&D'],'VIF_Score':[vif_admin,vif_market,vif_rnd]}
tab=pd.DataFrame(ds)
tab

#Second Model Creation
model2=smf.ols('Profit~rnd+admin+market',data=new).fit()
model2.summary()                  #Rsquare =0.963  and Adj Rsquare =0.961
pred=model2.predict(new)
pred.corr(new.Profit)             #0.9814917776544316
plt.plot(pred,);plt.plot(new.Profit,'ro')
sm.graphics.plot_partregress_grid(model2)
sm.graphics.influence_plot(model2)

rmse=sqrt(mean_squared_error(new.Profit, pred))
rmse                              #7180.335760510309

#Model 2 is best 




