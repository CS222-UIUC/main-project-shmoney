import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Explicitly set the API key and model path
NEWS_API_KEY = "f501f61f9d3449f885ffaf97eb23c506"
MODEL_PATH = r"C:\\Users\\aksah\\OneDrive\\Documents\\CS222\\main-project-shmoney\\keras_model.h5"

# Ensure model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at the path: {MODEL_PATH}")

# Load Keras Model
model = load_model(MODEL_PATH, compile=False)

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Streamlit App Title with Logo
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .header-container img {
        width: 50px;
        height: 50px;
    }
    </style>
    <div class="header-container">
        <h1>Comprehensive Stock Analysis & Prediction App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "Analyze stock trends, sentiment, and predictions with advanced visualizations.")

# Sidebar with Logo
st.sidebar.markdown(
    """
    <style>
    .sidebar-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .sidebar-logo img {
        width: 150px;
        height: auto;
    }
    </style>
    <div class="sidebar-logo">
        <img src="blob:https://manage.wix.com/47150275-e26e-445a-b445-5c12f8e63b43" alt="shmoney Logo">
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Settings
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-31"))
num_future_days = st.sidebar.slider(
    "Number of days to predict into the future", min_value=1, max_value=100, value=30
)
page = st.sidebar.radio(
    "Go to",
    ["Stock Data & Sentiment Analysis", "Moving Averages & Keras Predictions", "Prophet Model & Future Predictions"]
)

# Function to fetch news sentiment and articles
@st.cache_data
def fetch_news_sentiment(ticker):
    news = newsapi.get_everything(q=ticker, language="en", sort_by="relevancy")
    analyzer = SentimentIntensityAnalyzer()
    articles_info = []

    for article in news["articles"]:
        description = article.get("description", "")
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
                    "urlToImage": article.get("urlToImage"),
                }
            )
    return articles_info

# Main functionality
if st.sidebar.button("Run"):
    with st.spinner("Fetching data and performing analysis..."):
        # Fetch stock data
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)

        if page == "Stock Data & Sentiment Analysis":
            # Stock Data Summary
            st.subheader(f"Stock Data for {ticker}")
            st.write(df.describe())

            # News Sentiment Analysis
            st.subheader("News Sentiment Analysis")
            articles_info = fetch_news_sentiment(ticker)

            if articles_info:
                sentiment_df = pd.DataFrame(articles_info)
                sentiment_df["Date"] = pd.to_datetime(sentiment_df["publishedAt"]).dt.date

                # Ensure continuous dates in the sentiment data
                all_dates = pd.date_range(start=sentiment_df["Date"].min(), end=sentiment_df["Date"].max())
                daily_sentiment = (
                    sentiment_df.groupby("Date")["sentiment_score"]
                    .mean()
                    .reindex(all_dates, fill_value=0)
                    .reset_index()
                )
                daily_sentiment.columns = ["Date", "mean"]

                daily_sentiment["std"] = (
                    sentiment_df.groupby("Date")["sentiment_score"]
                    .std()
                    .reindex(all_dates, fill_value=0)
                    .values
                )

                # Sentiment Trend with Confidence Interval
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(
                    x="Date", y="mean", data=daily_sentiment, ax=ax, color="blue", label="Average Sentiment"
                )
                ax.fill_between(
                    daily_sentiment["Date"],
                    daily_sentiment["mean"] - daily_sentiment["std"],
                    daily_sentiment["mean"] + daily_sentiment["std"],
                    color="blue", alpha=0.2, label="Sentiment Variability"
                )
                ax.set_title("Daily Sentiment Trend with Confidence Interval")
                ax.set_xlabel("Date")
                ax.set_ylabel("Average Sentiment Score")
                ax.legend()
                fig.autofmt_xdate()
                st.pyplot(fig)

                # Display Articles with Thumbnails
                st.subheader("Highlighted Articles")
                for _, row in sentiment_df.head(3).iterrows():
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if row.get("urlToImage"):
                            st.image(row["urlToImage"], width=100)
                        else:
                            st.write("No Image Available")
                    with col2:
                        st.write(f"**Title:** {row['title']}")
                        st.write(f"**Description:** {row['description']}")
                        st.write(f"**Sentiment Score:** {row['sentiment_score']}")
                        st.write(f"[Read more]({row['url']})")
                        st.write("---")

                # Option to view more articles
                if len(articles_info) > 3:
                    with st.expander("See More Articles"):
                        for _, row in sentiment_df.iloc[3:].iterrows():
                            st.write(f"**Title:** {row['title']}")
                            st.write(f"**Description:** {row['description']}")
                            st.write(f"**Sentiment Score:** {row['sentiment_score']}")
                            st.write(f"[Read more]({row['url']})")
                            st.write("---")
            else:
                st.warning("No relevant articles found.")

        elif page == "Moving Averages & Keras Predictions":
            # Moving Averages
            st.subheader("Moving Averages (100 & 200 Days)")
            ma100 = df["Close"].rolling(100).mean()
            ma200 = df["Close"].rolling(200).mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df["Date"], df["Close"], label="Closing Price")
            ax.plot(df["Date"], ma100, label="100-Day MA")
            ax.plot(df["Date"], ma200, label="200-Day MA")
            ax.legend()
            st.pyplot(fig)

            # Keras Model Predictions
            st.subheader("Keras Model Predictions")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

            training_size = int(len(scaled_data) * 0.70)
            test_data = scaled_data[training_size - 100 :]
            x_test = [test_data[i - 100 : i] for i in range(100, len(test_data))]
            y_test = [test_data[i, 0] for i in range(100, len(test_data))]
            x_test, y_test = np.array(x_test), np.array(y_test)

            y_predicted = model.predict(x_test)
            y_predicted = scaler.inverse_transform(y_predicted)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["Date"].iloc[training_size:], y_test, label="Original")
            ax.plot(df["Date"].iloc[training_size:], y_predicted, label="Predicted")
            ax.legend()
            st.pyplot(fig)

        elif page == "Prophet Model & Future Predictions":
            # Prophet Model Predictions
            st.subheader("Prophet Model Predictions")
            prophet_df = df[["Date", "Close"]]
            prophet_df.columns = ["ds", "y"]

            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=num_future_days)
            forecast = m.predict(future)

            # Plot Predictions
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            # Forecast Components with Daily and Hour of Day Trends
            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

            # Daily and Hour of Day Trends
            st.subheader("Daily and Hour of Day Trends")
            daily_fig = m.plot(forecast, xlabel="Date", ylabel="Trend")
            st.pyplot(daily_fig)

            # Future Predictions Table
            st.subheader("Future Predictions")
            future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(num_future_days)
            future_forecast.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
            future_forecast.set_index("Date", inplace=True)
            st.write(future_forecast)
