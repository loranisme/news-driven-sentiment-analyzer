# News Driven Sentiment Analyzer

This project is a research-oriented quantitative trading framework built with Python. It combines financial news sentiment analysis, technical indicators, and machine learning to study short-horizon stock timing signals.

The system is designed as a single-name research model, with the strongest empirical results currently observed on NVDA.

## Key Features

•	Data Collection: Automatically gathers financial news and historical OHLCV price data.
•	Sentiment Analysis: Uses FinBERT to extract sentiment signals from financial news text.
•	Feature Engineering: Combines sentiment features with technical and market-regime indicators such as moving averages, RSI, MACD, ATR, volatility, and relative strength.
•	Model Training: Uses rolling-window machine learning to generate short-horizon directional probabilities.
•	Signal Filtering: Combines model output with trend, volatility, and market-regime conditions before entering trades.
•	Risk Management: Includes volatility-based sizing, ATR stop-loss, trailing stop, drawdown pause, and leverage constraints.
•	Backtesting Framework: Supports walk-forward evaluation, transaction cost modeling, and performance attribution.
•	Dual-Core Design: Includes both a balanced core and an aggressive core, which can be combined into a simple ensemble timing framework.


## Example Result
<img width="1512" height="629" alt="Screenshot 2026-04-14 at 00 58 02" src="https://github.com/user-attachments/assets/14efff56-e9d9-4af5-a0c6-80f51be7a169" />

<img width="935" height="704" alt="Screenshot 2026-04-14 at 00 57 09" src="https://github.com/user-attachments/assets/b6b08538-a1f3-41ff-9cfa-c2a2d8f39691" />

<img width="686" height="628" alt="Screenshot 2026-04-14 at 00 59 39" src="https://github.com/user-attachments/assets/a5c5304a-ca18-4c4f-ac4e-b83fb659037c" />

## How to Run

### Install Dependencies

Install the required Python packages for the project.

### Run the Main Program

```bash
python -m src.main
```
## Limitations
•	The framework is currently research-focused, not production-ready

•	Model complexity is still relatively high

•	Output probabilities are better interpreted as scores / conviction proxies than perfectly calibrated probabilities

•	Further validation is still needed through:

•	IC analysis

•	quantile return spread analysis

•	threshold sensitivity tests

•	ablation studies

