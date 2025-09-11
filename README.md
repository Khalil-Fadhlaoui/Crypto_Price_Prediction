# 📈 Cryptocurrency Price Prediction using Reddit Sentiment

![Python](https://img.shields.io/badge/python-3.9-blue.svg)![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)![License](https://img.shields.io/badge/license-MIT-green.svg)

This project demonstrates an end-to-end data science workflow to predict the hourly price of Ethereum (ETH/USDT) by leveraging public sentiment from the r/CryptoCurrency subreddit. It showcases the power of feature engineering with Natural Language Processing (NLP) and the dramatic performance improvement gained by moving from classical time-series models to non-linear machine learning models like XGBoost.

**The final XGBoost model successfully reduced the prediction error (RMSE) by 86.68% compared to the initial baseline.**

![XGBoost Predictions vs Actual]<img width="1108" height="587" alt="result " src="https://github.com/user-attachments/assets/0d89b329-2c1a-4611-97bc-86d6ffbd9192" />

*<p align="center">Note: You should replace the image path above with a screenshot of your final prediction plot!</p>*

---

## 🚀 The Problem

Financial markets are notoriously difficult to predict. While traditional models rely on historical price data, public sentiment, especially from influential online communities, has emerged as a potentially powerful, yet noisy, predictive signal.

This project seeks to answer the question: **Can we systematically extract sentiment from Reddit discussions to create a model that accurately forecasts cryptocurrency price movements?**

---

## 🔧 The Approach

This project followed an iterative and hypothesis-driven approach, beginning with a simple baseline and progressively building a more sophisticated solution.

### 1. Baseline Model: SARIMAX with Basic Features
*   **Hypothesis:** The upvote `score` of Reddit posts can serve as a simple proxy for market sentiment.
*   **Action:** A SARIMAX (Seasonal Auto-Regressive Integrated Moving Average with eXogenous factors) model was trained on historical ETH prices, using rolling averages of post scores as external features.
*   **Result:** The model failed to find a statistically significant relationship between post scores and price changes, resulting in a high **RMSE of $293.00**. This established our baseline performance.

### 2. Feature Engineering: NLP for Sentiment Analysis
*   **Hypothesis:** The actual *textual content* of the posts contains a much richer sentiment signal than simple upvote scores.
*   **Action:** Used the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** library to perform sentiment analysis on the title and body of over 40,000 Reddit posts, creating a new `sentiment` feature.
*   **Result:** A new SARIMAX model trained with rolling averages of this sentiment feature showed statistically significant coefficients (especially for 8-hour and 24-hour windows) and achieved a modest performance improvement, reducing the **RMSE to $282.06**. This proved the value of NLP-driven feature engineering.

### 3. Advanced Modeling: Capturing Non-Linearity with XGBoost
*   **Hypothesis:** The relationship between market sentiment and price is complex and non-linear, a pattern that a linear model like SARIMAX cannot capture.
*   **Action:** Developed a powerful **XGBoost (Extreme Gradient Boosting)** regression model. Engineered a new feature set including lagged values of price and sentiment to give the model a memory of past conditions.
*   **Result:** The XGBoost model dramatically outperformed the SARIMAX model, achieving an **RMSE of just $37.56**. This represented an **86.68% reduction in error** from our best statistical model, confirming that the underlying patterns are indeed non-linear.

---

## 📊 Key Results

| Model                     | Features Used         | RMSE      | Error as % of Avg. Price |
| ------------------------- | --------------------- | --------- | ------------------------ |
| Baseline SARIMAX          | Reddit Post `score`   | $293.00   | 8.50%                    |
| Sentiment-Aware SARIMAX   | VADER `sentiment`     | $282.06   | 8.17%                    |
| **Final XGBoost Model**   | **VADER & Lag Features** | **$37.56** | **~1.1%**                |

---

## 💻 Technologies Used

*   **Language:** Python 3
*   **Core Libraries:** Pandas, NumPy, Matplotlib
*   **Time-Series Analysis:** Statsmodels (for SARIMAX and stationarity tests)
*   **Machine Learning:** Scikit-learn, **XGBoost**
*   **Natural Language Processing:** VADER
*   **Environment:** Jupyter Notebook

---



---
