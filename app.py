import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
import joblib

# Load environment variables
load_dotenv()

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))

# Function to fetch news sentiment and articles (cached)
@st.cache_data
def fetch_news_sentiment(ticker):
    news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
    analyzer = SentimentIntensityAnalyzer()
    articles_info = []

    for article in news['articles']:
        description = article['description']
        if description:
            score = analyzer.polarity_scores(description)
            articles_info.append({
                'title': article['title'],
                'description': description,
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name'],
                'sentiment_score': score['compound']
            })

    return articles_info

# App Title and Layout
st.title("Comprehensive Stock Analysis & Prediction App")
st.markdown("Analyze stock trends, sentiment, and predictions with advanced visualizations.")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-31"))

# Load Keras Model without compiling to avoid warning
model_path = "/Users/aaditroychowdhury/Documents/CS 222/Main branch/main-project-shmoney/keras_model.h5"
model = load_model(model_path, compile=False)

# Run button
if st.sidebar.button("Run"):
    with st.spinner("Fetching data and performing analysis..."):
        # Fetch stock data
        df = yf.download(ticker, start=start_date, end=end_date)

        # Display stock data summary
        st.subheader(f"Stock Data for {ticker}")
        st.write(df.describe())

        # Sentiment Analysis
        st.subheader("News Sentiment Analysis")
        articles_info = fetch_news_sentiment(ticker)

        if articles_info:
            # Convert article data to DataFrame and aggregate by date
            sentiment_df = pd.DataFrame(articles_info)
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['publishedAt']).dt.date
            daily_sentiment = sentiment_df.groupby('Date')['sentiment_score'].agg(['mean', 'std']).reset_index()
            daily_sentiment.fillna(0, inplace=True)  # Fill any NaNs with 0 in case of missing data

            # Plot daily sentiment with CI and improved date formatting
            st.subheader("Daily Sentiment Trend with Confidence Interval")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(
                x="Date", y="mean", data=daily_sentiment,
                ax=ax, color="blue", label="Average Sentiment"
            )
            ax.fill_between(
                daily_sentiment['Date'],
                daily_sentiment['mean'] - daily_sentiment['std'],
                daily_sentiment['mean'] + daily_sentiment['std'],
                color="blue", alpha=0.2, label="Sentiment Variability"
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
            for article in articles_info[:num_articles_to_show]:  # Show first two articles
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Description:** {article['description']}")
                st.write(f"**Sentiment Score:** {article['sentiment_score']}")
                st.write(f"[Read more]({article['url']})")
                st.write("---")

            # Use expander to show additional articles
            if len(articles_info) > num_articles_to_show:
                with st.expander("See More Articles"):
                    for article in articles_info[num_articles_to_show:]:  # Show remaining articles
                        st.write(f"**Title:** {article['title']}")
                        st.write(f"**Description:** {article['description']}")
                        st.write(f"**Sentiment Score:** {article['sentiment_score']}")
                        st.write(f"[Read more]({article['url']})")
                        st.write("---")
        else:
            st.warning("No relevant articles found.")

        # Stock price visualization
        # st.subheader(f"Stock Price for {ticker}")
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.plot(df['Close'], label='Closing Price')
        # ax.set_title("Closing Price Over Time")
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Price")
        # ax.legend()
        # st.pyplot(fig)

        # Moving Averages
        st.subheader("Moving Averages (100 & 200 Days)")
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Close'], label='Closing Price', color='blue')
        ax.plot(ma100, label='100-Day MA', color='red')
        ax.plot(ma200, label='200-Day MA', color='green')
        ax.set_title("Closing Price with Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)


        # Prepare data for Keras model
        training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
        testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

        scaler = MinMaxScaler(feature_range=(0,1))
        training_array = scaler.fit_transform(training)

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

