# Business Forecasting Dashboard

A modern, interactive dashboard for business forecasting and analysis using Statsmodels and Streamlit.

## Features

- 📈 Sales forecasting using SARIMAX time series modeling
- 💰 P&L analysis and trends
- 📊 Interactive visualizations
- 🔍 Key business metrics and insights
- 📅 Monthly performance analysis
- 🗓️ Seasonal pattern detection

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Format

Your Excel file (`Travellers Cavern Data.xlsx`) should contain the following columns:
- MONTH (Date format)
- SALES (Numeric)
- EXPENSES (Numeric)
- SALARY (Numeric)
- P&L (Numeric)

## Running the Dashboard

1. Make sure your Excel file is in the same directory as the application
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. The dashboard will open in your default web browser

## Usage

- Use the sidebar to control the forecast period
- Interact with charts by hovering, zooming, and panning
- View detailed monthly breakdowns in the table at the bottom
- Analyze trends and patterns in your business performance

## About the Forecasting

- The dashboard uses SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) for time series forecasting
- For monthly data, the model automatically detects seasonal patterns
- Confidence intervals show the range of potential future values
- Moving averages help identify underlying trends

## Notes

- The dashboard is optimized for monthly data
- All monetary values are displayed in your local currency format 