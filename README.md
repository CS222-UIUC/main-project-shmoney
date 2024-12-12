# Comprehensive Stock Analysis & Prediction App
---
## Overview
This project is a Streamlit-based application designed to perform comprehensive stock analysis and predictions. The app integrates multiple functionalities such as historical data visualization, news sentiment analysis, moving averages, predictions using machine learning models, and future price forecasting with the Prophet library. The app features a clean and responsive user interface enhanced with light and dark themes.
---
## Key Features
### Stock Data Analysis
- Fetches historical stock data using the Yahoo Finance API.
- Visualizes stock trends, including 100-day and 200-day moving averages.
### News Sentiment Analysis
- Leverages the NewsAPI and VADER Sentiment Analysis to provide:
  - Aggregated daily sentiment scores with confidence intervals.
  - Collapsible sections for exploring detailed news articles.
### Price Prediction Using Keras
- Employs a pre-trained Keras model to predict stock prices:
  - Scales data using MinMaxScaler.
  - Splits data into training and test sets.
  - Displays predictions versus actual stock prices.
### Future Price Forecasting with Prophet
- Utilizes the Prophet library for:
  - Predicting stock prices for a user-defined future period.
  - Visualizing forecast trends and their components.
### Customizable User Interface
- Interactive settings sidebar to input:
  - Stock ticker symbol.
  - Start and end dates for data.
  - Number of future days for predictions.
### Themes and Styling
- Switchable light and dark themes using a custom CSS stylesheet for:
  - Buttons, inputs, and hover effects.
  - Responsive design for enhanced usability.
---
## Technologies Used
### Frontend and UI
- Streamlit: For creating an interactive web app.
- CSS: Custom styles for theming and responsiveness.
### Backend
- Yahoo Finance API: To fetch historical stock data.
- NewsAPI: To gather relevant news articles.
- VADER Sentiment Analyzer: For computing sentiment scores.
- Keras: For stock price prediction using a neural network model.
- Prophet: For time-series forecasting.
- Flask: For routing and full-stack integration.
### Libraries and Tools
- Matplotlib & Seaborn: For data visualization.
- Pandas & NumPy: For data manipulation and analysis.
- Scikit-learn: For data preprocessing.
- dotenv: For managing environment variables securely.
---
## Setup and Installation
### Clone the Repository
```bash
git clone https://github.com/CS222-UIUC/main-project-shmoney/tree/main/
cd your-project-directory
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Set Environment Variables
- Create a `.env` file with your `MODEL_PATH` and `NEWS_API_KEY`, the 'NEWS_API_KEY' is from the newsapi website generated when you create an account:
  ```env
  MODEL_PATH=path/to/your/keras_model.h5
  NEWS_API_KEY=your_news_api_key
  ```
### Run the App
```bash
jupyter notebook new_model.ipynb
streamlit run app.py
```
run the jupyter notebook before you run the application, make sure you download keras_model.h5 locally
---
## How to Use the App
### Steps
1. Launch the app by running the above command.
2. Use the sidebar to:
   - Enter the stock ticker symbol (e.g., `AAPL`).
   - Select the date range for historical data.
   - Adjust the number of days for future prediction.
3. Click **Run** to perform analysis and predictions.
---
### Stock Data Visualization
- Displays closing prices with moving averages.
### Sentiment Analysis
- Shows sentiment trends with confidence intervals.
### Predictions vs. Original Prices
- Compares actual and predicted prices.
### Future Forecasts
- Displays Prophet forecasts and components.
---
## Future Updates
- Add more technical indicators for stock analysis.
- Integrate additional machine learning models.
- Enable real-time stock updates.
## Group Member Roles
- Akshay: Did frontend development given his past projects on website development and integrated the front and back end together to make a clean web app
- Aadit: Was involved in the ML development as he has prior experience in developing and training machine learning models
- Yana & Alexia: Alexia and Yana will work on the backend as they have expertise in using frameworks such as Flask and have deployed applications using these libraries for their honors courses. Yana will also work on CI pipeline because she has experience working with unit and integration tests
---
## License
This project is licensed under the MIT License.