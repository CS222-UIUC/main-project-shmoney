import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib


start = '2010-01-01'
end = '2024-10-31'

st.title("Stock Prediction App")

user_input = st.text_input('Select stock ticker', 'AAPL')
df = yf.download(user_input, start, end)

st.subheader('2010-2024')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')

fig = plt.figure(figsize= (12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('MOVING AVERAGE 100 (R) AND 200 (G)')

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12, 6))
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)


training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
training_array = scaler.fit_transform(training)

model = load_model('/Users/aaditroychowdhury/Documents/CS 222/Main branch/main-project-shmoney/keras_model.h5')


past_100_days = training.tail(100)
final_df = pd.concat([past_100_days, testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    
    x_test.append(input_data[i - 100 : i])
    y_test.append(input_data[i, 0])
   
    
x_test, y_test = np.array(x_test), np.array(y_test)


# Now you can make predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final figure

st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize= (12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
