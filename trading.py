import numpy as np 
import pandas as pd 
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


name = "" + str(sys.argv[1])
#read the file
df = pd.read_csv(name)

df.set_index('Date', inplace=True)
df.head()

df[['Open', 'Close', 'High','Low','Close','Adj Close']].plot(figsize=(18,5))
plt.title("Stock Plot for Overall Period", fontsize=17)

a = df.sort_values(by='Close',ascending= False).head(5)
a['Close']

a = df.sort_values(by='Close',ascending= True).head(5)
a['Close']

sns.jointplot(x='Volume', y='Close', data=df, kind='reg')

df['Daily_returns'] = df['Close'].pct_change() #Pandas dataframe.pct_change() function calculates the percentage change between the current and a prior element.
df.tail()

df[df['Daily_returns']==df['Daily_returns'].max()]['Daily_returns']
df[df['Daily_returns']==df['Daily_returns'].min()]['Daily_returns']

plt.figure(figsize=(15,5))
df['Daily_returns'].plot()
plt.xlabel("Date")
plt.ylabel("Percent")
plt.title("Stock Daily Returns",fontsize= 15 )

sns.set_style('whitegrid')
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
sns.histplot(data= df['Daily_returns'], bins=60)
ax1.set_xlabel("Daily returns %")
ax1.set_ylabel("Percent")
ax1.set_title("Stock Daily Returns Percentage",fontsize= 15 )
ax1.text(-0.18,60,"Extreme Low\n Returns",fontsize= 12)
ax1.text(0.09,60,"Extreme High\n Returns", fontsize= 12)
plt.show()

df['Cum_returns']= (df['Daily_returns']+1).cumprod()
df.head()

sns.set_style('whitegrid')
plt.figure(figsize=(15,5))
df['Cum_returns'].plot()
plt.xlabel("Date")
plt.ylabel("Percent")
plt.title("Stock Cumulative Returns",fontsize= 15 )
plt.legend()

sns.set_style('whitegrid')
f= plt.figure(figsize=(12,5))
df['Close'].loc['2021-01-04': '2022-03-21'].rolling(window=30).mean().plot(label='30 Day Avg')
df['Close'].loc['2021-01-04':'2022-03-21'].plot(label='CLOSE price')
plt.title(" Comparison of the moving average & Close price", fontsize=17)
plt.legend()

f= plt.figure(figsize=(12,5))
df['Close'].rolling(window=30).mean().plot(label='30 Day Avg')
df['Close'].plot(label='CLOSE price')
plt.title(" Comparison of the moving average & Close price for Overall Period", fontsize=17)
plt.legend()

df['Mean Avg 30Day'] = df['Close'].rolling(window=30).mean() # MA= mean Average
df['STD 30Day'] = df['Close'].rolling(window=30).std()

df['Upper Band']= df['Mean Avg 30Day'] + (df['STD 30Day'] *2)
df['Lower Band']= df['Mean Avg 30Day'] - (df['STD 30Day'] *2)
df.head()

df[['Adj Close', 'Mean Avg 30Day', 'Upper Band', 'Lower Band']].plot(figsize=(18,5))
plt.title(" Bollinger Band Plot for Overall Period", fontsize=17)

X= df[['Open', 'High', 'Low', 'Close', 'Volume']]
y= df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=10)
model.fit(X_train, y_train)
predict = model.predict(X_test)

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(X_train, y_train) * 100:.2f}% and Test Score : {model.score(X_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 

predictions = pd.DataFrame({"Predictions": predict}, index=pd.date_range(start=df.index[-1], periods=len(predict), freq="D"))

#collecting future days from predicted values
days_df = pd.DataFrame(predictions[3:10])

print(days_df)
