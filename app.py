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

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-31"))

num_future_days = st.sidebar.slider(
    "Number of days to predict into the future", min_value=1, max_value=100, value=30
)
page = st.sidebar.radio("Go to", ["Stock Data & Sentiment Analysis", "Moving Average & Keras Model Predictions", "Prophet Model & Future Predictions"])


# Load Keras Model without compiling to avoid warning
model_path = os.getenv("MODEL_PATH")

# Load the Keras model
model = load_model(model_path, compile=False)

# Check if the file exists

        # Fetch stock data (cached to prevent repeated downloads)
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
           return yf.download(ticker, start=start_date, end=end_date)

        # Fetch stock data
df = fetch_stock_data(ticker, start_date, end_date)

        # Reset index to add Date column
df.reset_index(inplace=True)

        # Ensure Date column exists and is in datetime format
if "Date" not in df.columns:
            raise KeyError("The 'Date' column is missing from the DataFrame.")

        # Display stock data summary
st.subheader(f"Stock Data for {ticker}")
st.write(df.describe())

        # Sentiment Analysis
if page == "Stock Data & Sentiment Analysis":
            st.subheader("News Sentiment Analysis")
            articles_info = fetch_news_sentiment(ticker)

            if articles_info:
                # Convert article data to DataFrame and aggregate by date
                sentiment_df = pd.DataFrame(articles_info)
                sentiment_df["Date"] = pd.to_datetime(sentiment_df["publishedAt"]).dt.date
                daily_sentiment = (
                    sentiment_df.groupby("Date")["sentiment_score"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                daily_sentiment.fillna(
                    0, inplace=True
                )  # Fill any NaNs with 0 in case of missing data

                # Plot daily sentiment with CI and improved date formatting
                st.subheader("Daily Sentiment Trend with Confidence Interval")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(
                    x="Date",
                    y="mean",
                    data=daily_sentiment,
                    ax=ax,
                    color="blue",
                    label="Average Sentiment",
                )
                ax.fill_between(
                    daily_sentiment["Date"],
                    daily_sentiment["mean"] - daily_sentiment["std"],
                    daily_sentiment["mean"] + daily_sentiment["std"],
                    color="blue",
                    alpha=0.2,
                    label="Sentiment Variability",
                )
                ax.set_title(f"Sentiment Trend for {ticker}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Average Sentiment Score")
                ax.legend()
                fig.autofmt_xdate()  # Rotate dates for better readability
                st.pyplot(fig)

                # Display two articles with collapsible option for more articles
                st.subheader("Recent News Articles")
                num_articles_to_show = 2
                for article in articles_info[
                    :num_articles_to_show
                ]:  # Show first two articles
                    st.write(f"**Title:** {article['title']}")
                    st.write(f"**Description:** {article['description']}")
                    st.write(f"**Sentiment Score:** {article['sentiment_score']}")
                    st.write(f"[Read more]({article['url']})")
                    st.write("---")

                # Use expander to show additional articles
                if len(articles_info) > num_articles_to_show:
                    with st.expander("See More Articles"):
                        for article in articles_info[
                            num_articles_to_show:
                        ]:  # Show remaining articles
                            st.write(f"**Title:** {article['title']}")
                            st.write(f"**Description:** {article['description']}")
                            st.write(f"**Sentiment Score:** {article['sentiment_score']}")
                            st.write(f"[Read more]({article['url']})")
                            st.write("---")
            else:
                st.warning("No relevant articles found.")

            # --------------------------------------------------------------------------

            # Ensure the index is reset to include a Date column if necessary
            if "Date" not in df.columns:
                df.reset_index(inplace=True)
elif page == "Moving Average & Keras Model Predictions":
            # Moving Averages
            st.subheader("Moving Averages (100 & 200 Days)")
            ma100 = df["Close"].rolling(100).mean()
            ma200 = df["Close"].rolling(200).mean()
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot with actual dates on x-axis
            ax.plot(
                df["Date"], df["Close"], label="Closing Price", color="#000000"
            )  # Black for Closing Price
            ax.plot(
                df["Date"], ma100, label="100-Day MA", color="#2ca02c"
            )  # Green for 100-Day MA
            ax.plot(
                df["Date"], ma200, label="200-Day MA", color="#ff7f0e"
            )  # Orange for 200-Day MA

            ax.set_title("Closing Price with Moving Averages")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            fig.autofmt_xdate()  # Automatically format date labels for better readability
            st.pyplot(fig)
            # --------------------------------------------------------------------------

            # Prepare data for Keras model
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

            # Split data into training and testing sets
            training_size = int(len(scaled_data) * 0.70)
            train_data = scaled_data[0:training_size]
            test_data = scaled_data[training_size - 100 :]

            # Create test dataset
            x_test = []
            y_test = []

            for i in range(100, len(test_data)):
                x_test.append(test_data[i - 100 : i])
                y_test.append(test_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions using the model
            y_predicted = model.predict(x_test)

            # Inverse transform to get actual prices
            y_predicted = scaler.inverse_transform(y_predicted)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Ensure Date column exists in your DataFrame
            if "Date" not in df.columns:
                raise KeyError("The 'Date' column is missing from the DataFrame.")

            # Get the corresponding dates for the test data
            test_dates = df["Date"].iloc[training_size + 100 :]  # Match indices with y_test

            # Ensure test_dates and y_test have the same dimensions
            y_test = y_test[: len(test_dates)]  # Slice y_test to match test_dates
            y_predicted = y_predicted[
                : len(test_dates)
            ]  # Slice y_predicted for consistency
            # Final figure with updated x-axis
            st.subheader("Predictions vs Original")
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(
                test_dates, y_test, color="#000000", label="Original Price"
            )  # Black for original price
            plt.plot(
                test_dates, y_predicted, color="#1f77b4", alpha=0.7, label="Predicted Price"
            )  # Adjust transparency
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.xticks(rotation=45)  # Rotate x-axis labels for readability
            plt.legend()
            st.pyplot(fig2)

            # ------------------------------------------------------------------------------------------
elif page == "Prophet Model & Future Predictions":
            # Use Prophet for future prediction

            # Prepare data for Prophet
            prophet_df = df.reset_index()[["Date", "Close"]]
            prophet_df.columns = ["ds", "y"]

            # Remove timezone information from 'ds' column if necessary
            if prophet_df["ds"].dt.tz is not None:
                prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

            # Initialize and fit the model
            m = Prophet(daily_seasonality=True)
            m.fit(prophet_df)

            # Create future dataframe with business days frequency
            future = m.make_future_dataframe(periods=num_future_days, freq="B")

            # Predict future prices
            forecast = m.predict(future)

            # Plot the forecast
            st.subheader(f"Future Price Prediction for {ticker} using Prophet")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)

            # Plot the forecast components
            st.subheader("Forecast Components")
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

            # Display the forecasted data
            st.subheader("Future Predictions")

            # Extract and rename columns
            future_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(
                num_future_days
            )
            future_forecast.rename(
                columns={
                    "ds": "Date",
                    "yhat": "Predicted Price",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound",
                },
                inplace=True,
            )
            future_forecast.set_index("Date", inplace=True)

            future_forecast.index = future_forecast.index.date

            # Display renamed DataFrame
            st.write(future_forecast)