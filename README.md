"# main-project-shmoney"

Comprehensive Stock Analysis & Prediction App

Overview
This project is a Streamlit-based application designed to perform comprehensive stock analysis and predictions. The app integrates multiple functionalities such as historical data visualization, news sentiment analysis, moving averages, predictions using machine learning models, and future price forecasting with the Prophet library. The app features a clean and responsive user interface enhanced with light and dark themes.

Key Features:
1. Stock Data Analysis
- Fetches historical stock data using the Yahoo Finance API.
- Visualizes stock trends including 100-day and 200-day moving averages**.

2. News Sentiment Analysis
- Leverages the NewsAPI and VADER Sentiment Analysis to provide:
  - Aggregated daily sentiment scores with confidence intervals.
  - Collapsible sections for exploring detailed news articles.

3. Price Prediction Using Keras
- Employs a pre-trained Keras model to predict stock prices:
  - Scales data using MinMaxScaler.
  - Splits data into training and test sets.
  - Displays predictions versus actual stock prices.

4. Future Price Forecasting with Prophet
- Utilizes the Prophet library for:
  - Predicting stock prices for a user-defined future period.
  - Visualizing forecast trends and their components.

5. Customizable User Interface
- Interactive settings sidebar to input:
  - Stock ticker symbol.
  - Start and end dates for data.
  - Number of future days for predictions.

6. Themes and Styling
- Switchable light and dark themes using a custom CSS stylesheet for:
  - Buttons, inputs, and hover effects.
  - Responsive design for enhanced usability.

---

Technologies Used:
Frontend and UI
- Streamlit: For creating an interactive web app.
- CSS: Custom styles for theming and responsiveness.

Backend
- Yahoo Finance API: To fetch historical stock data.
- NewsAPI: To gather relevant news articles.
- VADER Sentiment Analyzer: For computing sentiment scores.
- Keras: For stock price prediction using a neural network model.
- Prophet: For time-series forecasting.
-Flask: Routing, fullstack integration

Libraries and Tools
- Matplotlib & Seaborn: For data visualization.
- Pandas & NumPy: For data manipulation and analysis.
- Scikit-learn: For data preprocessing.
- dotenv: For managing environment variables securely.

---
Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CS222-UIUC/main-project-shmoney/tree/main/
   cd your-project-directory
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   - Create a `.env` file with your `MODEL_PATH` and `NEWS_API_KEY`.
   ```env
   MODEL_PATH=path/to/your/keras_model.h5
   NEWS_API_KEY=your_news_api_key
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

How to Use the app
1. Launch the app by running the above command.
2. Use the sidebar to:
   - Enter the stock ticker symbol (e.g., `AAPL`).
   - Select the date range for historical data.
   - Adjust the number of days for future prediction.
3. Click Run to perform analysis and predictions.

---

Screenshots
1. Stock Data Visualization
- Displays closing prices with moving averages.

2. Sentiment Analysis
- Shows sentiment trends with confidence intervals.

3. Predictions vs Original Prices
- Compares actual and predicted prices.

4. Future Forecasts
- Displays Prophet forecasts and components.

---

Future Updates
- Add more technical indicators for stock analysis.
- Integrate additional machine learning models.
- Enable real-time stock updates.

---

## License
This project is licensed under the MIT License.