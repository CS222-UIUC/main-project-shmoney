import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))


# Caching function to load Keras Model once
@st.cache_resource
def load_keras_model():
    model_path = os.getenv("MODEL_PATH")
    model = load_model(model_path, compile=False)
    return model

# Caching function to fetch stock data once
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

# Function to fetch news sentiment and articles (cached)
@st.cache_data
def fetch_news_sentiment(ticker):
    news = newsapi.get_everything(q=ticker, language="en", sort_by="relevancy")
    analyzer = SentimentIntensityAnalyzer()
    articles_info = []

    for article in news["articles"]:
        description = article["description"]
        if description:
            score = analyzer.polarity_scores(description)
            articles_info.append(
                {
                    "title": article["title"],
                    "description": description,
                    "url": article["url"],
                    "publishedAt": article["publishedAt"],
                    "source": article["source"]["name"],
                    "sentiment_score": score["compound"],
                }
            )

    return articles_info

# App Title and Layout
st.title("Comprehensive Stock Analysis & Prediction App")
st.markdown(
    "Analyze stock trends, sentiment, and predictions with advanced visualizations."
)

# Sidebar settings
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-31"))

num_future_days = st.sidebar.slider(
    "Number of days to predict into the future", min_value=1, max_value=100, value=30
)
page = st.sidebar.radio("Go to", ["Stock Data & Sentiment Analysis", "Moving Average & Keras Model Predictions", "Prophet Model & Future Predictions"])

# Load the Keras model
model = load_keras_model()

# Fetch stock data once
df = fetch_stock_data(ticker, start_date, end_date)

# Display stock data summary
st.subheader(f"Stock Data for {ticker}")
st.write(df.describe())

# Handling the different pages
if page == "Stock Data & Sentiment Analysis":
    st.subheader("News Sentiment Analysis")
    articles_info = fetch_news_sentiment(ticker)

    if articles_info:
        sentiment_df = pd.DataFrame(articles_info)
        sentiment_df["Date"] = pd.to_datetime(sentiment_df["publishedAt"]).dt.date
        daily_sentiment = sentiment_df.groupby("Date")["sentiment_score"].agg(["mean", "std"]).reset_index()
        daily_sentiment.fillna(0, inplace=True)

        # Plot sentiment trend
        st.subheader("Daily Sentiment Trend with Confidence Interval")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x="Date", y="mean", data=daily_sentiment, ax=ax, color="blue", label="Average Sentiment")
        ax.fill_between(
            daily_sentiment["Date"], daily_sentiment["mean"] - daily_sentiment["std"], daily_sentiment["mean"] + daily_sentiment["std"], color="blue", alpha=0.2, label="Sentiment Variability"
        )
        ax.set_title(f"Sentiment Trend for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Sentiment Score")
        ax.legend()
        fig.autofmt_xdate()  # Rotate dates for better readability
        st.pyplot(fig)

        # Display articles
        st.subheader("Recent News Articles")
        num_articles_to_show = 2
        for article in articles_info[:num_articles_to_show]:
            st.write(f"**Title:** {article['title']}")
            st.write(f"**Description:** {article['description']}")
            st.write(f"**Sentiment Score:** {article['sentiment_score']}")
            st.write(f"[Read more]({article['url']})")
            st.write("---")

        # Use expander to show additional articles
        if len(articles_info) > num_articles_to_show:
            with st.expander("See More Articles"):
                for article in articles_info[num_articles_to_show:]:
                    st.write(f"**Title:** {article['title']}")
                    st.write(f"**Description:** {article['description']}")
                    st.write(f"**Sentiment Score:** {article['sentiment_score']}")
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")

elif page == "Moving Average & Keras Model Predictions":
    # Moving averages and Keras Model predictions
    st.subheader("Moving Averages (100 & 200 Days)")
    ma100 = df["Close"].rolling(100).mean()
    ma200 = df["Close"].rolling(200).mean()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["Date"], df["Close"], label="Closing Price", color="#000000")
    ax.plot(df["Date"], ma100, label="100-Day MA", color="#2ca02c")
    ax.plot(df["Date"], ma200, label="200-Day MA", color="#ff7f0e")

    ax.set_title("Closing Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()  # Automatically format date labels for better readability
    st.pyplot(fig)

    # Prepare data for Keras model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    # Split data into training and testing sets
    training_size = int(len(scaled_data) * 0.70)
    train_data = scaled_data[:training_size]
    test_data = scaled_data[training_size - 100:]

    # Create test dataset
    x_test, y_test = [], []
    for i in range(100, len(test_data)):
        x_test.append(test_data[i - 100:i])
        y_test.append(test_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions using the model
    y_predicted = model.predict(x_test)

    # Inverse transform to get actual prices
    y_predicted = scaler.inverse_transform(y_predicted)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

   
   # Ensure y_test is flattened to shape (1120,)
    y_test = y_test.flatten()

    # Check if y_test and test_dates need to match
    num_predictions = len(y_test)

    # Slice test_dates to match the number of predictions (y_test length)
    test_dates = df["Date"].iloc[training_size:training_size + num_predictions].reset_index(drop=True)

    # Ensure the lengths of both arrays match
    assert len(test_dates) == len(y_test), f"Length mismatch: test_dates({len(test_dates)}) != y_test({len(y_test)})"

    # Plot predictions vs original
    st.subheader("Predictions vs Original")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, color="#000000", label="Original Price")
    plt.plot(test_dates, y_predicted, color="#1f77b4", alpha=0.7, label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(fig2)
    
elif page == "Prophet Model & Future Predictions":
    # Reset the index to convert the multi-level index into columns
    df_reset = df.reset_index()
    
    # Rename the columns for easier access
    df_reset.columns = ['Price', 'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Drop rows with missing 'Date' values
    df_reset = df_reset.dropna(subset=['Date'])

    # Convert 'Date' to datetime format and remove timezone (make it timezone-naive)
    df_reset['Date'] = pd.to_datetime(df_reset['Date'], errors='coerce').dt.tz_localize(None)

    # Drop any rows where 'Date' couldn't be converted (NaT)
    df_reset = df_reset.dropna(subset=['Date'])

    # Create the prophet_df DataFrame with 'Date' and 'Close'
    prophet_df = df_reset[["Date", "Close"]].reset_index(drop=True)

    # If prophet_df has more than 2 columns, trim it to only the first two
    if prophet_df.shape[1] > 2:
        prophet_df = prophet_df.iloc[:, :2]  

    # Rename the columns to match Prophet's expected format
    prophet_df.columns = ["ds", "y"]

    # Initialize and fit the Prophet model
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)


    # Create future dataframe
    future = m.make_future_dataframe(periods=num_future_days, freq="B")
    forecast = m.predict(future)

    # Plot forecast
    st.subheader(f"Future Price Prediction for {ticker} using Prophet")
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

    # Plot forecast components
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    # Display forecasted data
    st.subheader("Future Predictions")
    future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(num_future_days)
    future_forecast.rename(columns={"ds": "Date", "yhat": "Predicted Price", "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"}, inplace=True)
    future_forecast.set_index("Date", inplace=True)
    st.write(future_forecast)
