import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Set the date range for data fetching
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title('Stock Forecast App (ARIMA)')

# Stock selection and prediction period in years
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 252  # Number of trading days in a year

# Load data function with caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data function
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode='lines'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode='lines'))
    fig.update_layout(
        title_text='Time Series Data with Rangeslider',
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for ARIMA model (using only 'Close' prices)
df_train = data[['Date', 'Close']].copy()
df_train.set_index('Date', inplace=True)

# Debug: Check df_train contents and data types
st.write("Historical data (df_train):")
st.write(df_train.tail())
st.write("Data types in df_train:", df_train.dtypes)

# Fit ARIMA model
st.write("Training ARIMA model...")
model = ARIMA(df_train['Close'], order=(5, 1, 0))  # ARIMA(p,d,q)
model_fit = model.fit()

# Forecast for the future period
forecast = model_fit.forecast(steps=period)
forecast_dates = pd.date_range(df_train.index[-1] + pd.Timedelta(days=1), periods=period, freq='B')

# Prepare DataFrame for plotting forecast
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
forecast_df.set_index('Date', inplace=True)

# Debug: Check forecast_df contents and data types
st.write("Forecast data (forecast_df):")
st.write(forecast_df.tail())
st.write("Data types in forecast_df:", forecast_df.dtypes)

# Plot forecast data
st.subheader('Forecast data')
st.write(forecast_df.tail())

def plot_forecast_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Close'], name="Historical Data", mode='lines'))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name="Forecast", mode='lines'))
    fig.update_layout(
        title_text=f'Forecast plot for {n_years} years',
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

plot_forecast_data()
