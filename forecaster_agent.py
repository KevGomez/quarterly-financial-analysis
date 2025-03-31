import os
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import acf, pacf
import itertools
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

llm = OpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"])

# Example data for financial metrics (dummy data) - kept as fallback
# sample_data = {
#     "REXP.N0000": {
#         "revenue": [1250000, 1340000, 1420000, 1380000, 1460000, 1520000, 1580000, 1640000],
#         "net_income": [125000, 142000, 156000, 138000, 152000, 164000, 172000, 178000],
#         "net_profit": [125000, 142000, 156000, 138000, 152000, 164000, 172000, 178000],
#         "basic_eps": [10.2, 12.5, 13.6, 11.8, 12.9, 13.7, 14.5, 15.2]
#     },
#     "DIPD.N0000": {
#         "revenue": [850000, 920000, 980000, 1020000, 1080000, 1150000, 1210000, 1280000],
#         "net_income": [85000, 94000, 102000, 106000, 112000, 118000, 125000, 132000],
#         "net_profit": [85000, 94000, 102000, 106000, 112000, 118000, 125000, 132000],
#         "basic_eps": [7.1, 8.2, 8.7, 9.2, 9.5, 10.1, 10.8, 11.3]
#     }
# }

# Mapping of UI terms to JSON metric names
METRIC_MAPPINGS = {
    "Revenue": "revenue",
    "Net Profit": "net_profit",
    "Net Income": "net_income",
    "EPS": "basic_eps",
    "Revenue Trend": "revenue",
    "EPS movement": "basic_eps",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
}

def get_financial_data(company: str, metric: str) -> List[float]:
    """
    Get historical financial data for a specific company and metric from JSON files.
    
    Args:
        company: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        
    Returns:
        List of historical values for the specified metric in chronological order
    """
    # Determine the dataset folder based on company
    folder_name = "REXP_Datasets" if company.startswith("REXP") else "DIPD_Datasets"
    
    # Path to look for JSON files
    json_pattern = os.path.join(folder_name, "*.json")
    
    # Get list of all JSON files in the directory
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {folder_name}. Using sample data as fallback.")
        # Fall back to sample data if no files found
        # if company in sample_data and metric in sample_data[company]:
        #     return sample_data[company][metric]
        return []
    
    # Load data from all JSON files
    quarterly_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check if the metric exists in the data
                if metric.lower() in data:
                    # Create a tuple with (year, quarter, value) for sorting
                    year = data.get('year', '0000')
                    quarter = data.get('quarter', 'Q0')
                    value = data[metric.lower()]
                    
                    quarterly_data.append((year, quarter, value))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if not quarterly_data:
        print(f"No data for metric '{metric}' found in {folder_name}. Using sample data as fallback.")
        # Fall back to sample data if no metric data found
        # if company in sample_data and metric in sample_data[company]:
        #     return sample_data[company][metric]
        return []
    
    # Sort by year and quarter
    quarterly_data.sort(key=lambda x: (x[0], x[1]))
    
    # Extract just the values in chronological order
    return [float(item[2]) for item in quarterly_data]

def get_metric_timeseries(company_code: str, metric: str) -> Tuple[pd.Series, List[str]]:
    """
    Get time-indexed series for a specific company and metric.
    
    Args:
        company_code: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        
    Returns:
        Tuple containing a pandas Series with time index and list of quarter labels
    """
    # Map user-friendly metric names to JSON keys
    if metric in METRIC_MAPPINGS:
        metric = METRIC_MAPPINGS[metric]
    
    # Get the raw data
    data = get_financial_data(company_code, metric)
    
    if not data:
        return pd.Series(dtype=float), []
    
    # Determine the dataset folder based on company
    folder_name = "REXP_Datasets" if company_code.startswith("REXP") else "DIPD_Datasets"
    json_files = glob.glob(os.path.join(folder_name, "*.json"))
    
    # Create a list of period labels (e.g., Q1 2022, Q2 2022, etc.)
    quarters = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                if 'quarter' in file_data and 'year' in file_data:
                    quarters.append((file_data['year'], file_data['quarter'], f"{file_data['quarter']} {file_data['year']}"))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Sort by year and quarter
    quarters.sort(key=lambda x: (x[0], x[1]))
    
    # Create labels
    period_labels = [item[2] for item in quarters]
    
    # Create a pandas Series with the period labels as the index
    if len(data) == len(period_labels):
        series = pd.Series(data, index=period_labels)
        return series, period_labels
    else:
        # In case of mismatched lengths (should not happen), default to numeric index
        print(f"Warning: Mismatched lengths - data: {len(data)}, labels: {len(period_labels)}")
        series = pd.Series(data)
        default_labels = [f"Period {i+1}" for i in range(len(data))]
        return series, default_labels

def select_optimal_arima_order(series: pd.Series) -> Tuple[int, int, int]:
    """
    Select optimal ARIMA order (p,d,q) using AIC criterion.
    
    Args:
        series: Time series data
        
    Returns:
        Tuple (p,d,q) with the optimal ARIMA order
    """
    # If series is too short, return a simple model
    if len(series) < 8:
        return (1, 0, 0)  # Simple AR(1) model for short series
    
    # Check if first difference is needed (d=1)
    # This is a simplified approach
    d = 1 if (series.diff().dropna().std() < series.std()) else 0
    
    # Set range of p and q based on available data
    p_max = min(3, (len(series) - 1) // 4)
    q_max = min(3, (len(series) - 1) // 4)
    
    # Initialize variables for best model
    best_aic = float('inf')
    best_order = (0, d, 0)
    
    # Try various combinations of p and q
    for p, q in itertools.product(range(p_max + 1), range(q_max + 1)):
        if p == 0 and q == 0:
            continue  # Skip ARIMA(0,d,0)
        
        try:
            model = ARIMA(series, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except Exception as e:
            continue  # Skip problematic model specifications
    
    return best_order

def run_arima_forecast(series: pd.Series, steps: int = 4) -> Dict[str, Any]:
    """
    Run ARIMA forecast with confidence intervals.
    
    Args:
        series: Time series data
        steps: Number of steps to forecast
        
    Returns:
        Dictionary with forecast results including confidence intervals
    """
    if len(series) < 3:
        return {
            "error": f"Not enough data points for ARIMA model. Need at least 3, but got {len(series)}."
        }
    
    # Choose optimal ARIMA order
    order = select_optimal_arima_order(series)
    
    # Fit ARIMA model
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    
    # Generate forecast
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
    
    # Prepare the result
    result = {
        "forecast": forecast_mean.tolist(),
        "lower_ci": forecast_ci.iloc[:, 0].tolist(),
        "upper_ci": forecast_ci.iloc[:, 1].tolist(),
        "model_order": order,
        "method": "arima",
        "aic": fitted_model.aic,
        "bic": fitted_model.bic
    }
    
    return result

def plot_forecast(series: pd.Series, forecast_result: Dict[str, Any], period_labels: List[str], 
                  metric: str, company: str) -> Figure:
    """
    Create a plot of the forecast with confidence intervals.
    
    Args:
        series: Historical time series data
        forecast_result: Dictionary with forecast results
        period_labels: Labels for the time periods
        metric: Metric name for display
        company: Company name
        
    Returns:
        Matplotlib figure
    """
    # Create future period labels (Q1, Q2, etc.)
    last_period = period_labels[-1] if period_labels else "Last Period"
    parts = last_period.split()
    
    if len(parts) >= 2:
        quarter, year = parts[0], parts[1]
        # Map quarter to number
        q_num = int(quarter[1])
        
        future_labels = []
        current_year = int(year)
        current_q = q_num
        
        for _ in range(len(forecast_result["forecast"])):
            current_q += 1
            if current_q > 4:
                current_q = 1
                current_year += 1
            future_labels.append(f"Q{current_q} {current_year}")
    else:
        # Fallback if we can't parse the period format
        future_labels = [f"Future {i+1}" for i in range(len(forecast_result["forecast"]))]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical data
    x_hist = range(len(series))
    ax.plot(x_hist, series.values.tolist(), 'b-o', label="Historical")
    
    # Plot forecast
    x_forecast = range(len(series) - 1, len(series) + len(forecast_result["forecast"]))
    ax.plot(x_forecast, [series.iloc[-1]] + forecast_result["forecast"], 'r-o', label="Forecast")
    
    # Plot confidence intervals
    ax.fill_between(
        range(len(series), len(series) + len(forecast_result["forecast"])),
        forecast_result["lower_ci"],
        forecast_result["upper_ci"],
        color='r',
        alpha=0.2,
        label="95% Confidence Interval"
    )
    
    # Set labels with all period labels and future labels
    all_labels = period_labels + future_labels
    
    # If we have too many labels, show only some of them
    if len(all_labels) > 10:
        stride = max(1, len(all_labels) // 10)
        display_labels = all_labels[::stride]
        display_positions = range(0, len(all_labels), stride)
        
        ax.set_xticks(display_positions)
        ax.set_xticklabels(display_labels, rotation=45)
    else:
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45)
    
    # Add labels and title
    ax.set_xlabel("Quarter")
    ax.set_ylabel(f"{metric} (LKR '000)")
    ax.set_title(f"{company} - {metric} Forecast")
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def time_series_forecast(company: str, metric: str, periods: int = 1) -> Dict[str, Any]:
    """
    Predict future values using basic time series trend analysis.
    
    Args:
        company: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    data = get_financial_data(company, metric)
    if not data:
        return {"error": f"No data available for {company}, metric: {metric}"}
    
    data_series = pd.Series(data)
    trend = np.polyfit(range(len(data_series)), data_series, 1)
    forecast = [trend[0] * (len(data_series) + i) + trend[1] for i in range(1, periods + 1)]
    
    return {
        "company": company,
        "metric": metric,
        "historical_data": data,
        "forecast": forecast,
        "method": "time_series"
    }

def moving_average_forecast(company: str, metric: str, window: int = 3, periods: int = 1) -> Dict[str, Any]:
    """
    Predict future values using simple moving average.
    
    Args:
        company: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        window: Number of periods for moving average
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    data = get_financial_data(company, metric)
    if not data:
        return {"error": f"No data available for {company}, metric: {metric}"}
    
    if len(data) < window:
        window = len(data)  # Adjust window if not enough data
        
    data_series = pd.Series(data)
    moving_avg = data_series.rolling(window=window).mean().iloc[-1]
    forecast = [moving_avg] * periods
    
    return {
        "company": company,
        "metric": metric,
        "historical_data": data,
        "forecast": forecast,
        "method": "moving_average"
    }

def exponential_smoothing_forecast(company: str, metric: str, periods: int = 1) -> Dict[str, Any]:
    """
    Predict future values using Exponential Smoothing.
    
    Args:
        company: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    data = get_financial_data(company, metric)
    if not data:
        return {"error": f"No data available for {company}, metric: {metric}"}
    
    data_series = pd.Series(data)
    
    # Adjust settings based on data length
    if len(data) >= 8:  # Need enough data for seasonal patterns
        model = ExponentialSmoothing(data_series, seasonal_periods=4)
    else:
        model = ExponentialSmoothing(data_series)
        
    fitted_model = model.fit()
    forecast = fitted_model.forecast(periods).tolist()
    
    return {
        "company": company,
        "metric": metric,
        "historical_data": data,
        "forecast": forecast,
        "method": "exponential_smoothing"
    }

def arima_forecast(company: str, metric: str, periods: int = 1) -> Dict[str, Any]:
    """
    Predict future values using ARIMA model.
    
    Args:
        company: Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')
        metric: Financial metric (e.g., 'revenue', 'net_income')
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    # Get the data as a series with period labels
    data_series, period_labels = get_metric_timeseries(company, metric)
    
    if len(data_series) == 0:
        return {"error": f"No data available for {company}, metric: {metric}"}
    
    # Need at least 3 data points for ARIMA
    if len(data_series) < 3:
        return {
            "error": f"Not enough data points for ARIMA model. Need at least 3, but got {len(data_series)}.",
            "company": company,
            "metric": metric,
            "historical_data": data_series.tolist()
        }
    
    # Run the ARIMA forecast
    forecast_result = run_arima_forecast(data_series, steps=periods)
    
    if "error" in forecast_result:
        forecast_result["company"] = company
        forecast_result["metric"] = metric
        forecast_result["historical_data"] = data_series.tolist() 
        return forecast_result
    
    # Add additional information to the result
    forecast_result["company"] = company
    forecast_result["metric"] = metric
    forecast_result["historical_data"] = data_series.tolist()
    forecast_result["period_labels"] = period_labels
    
    return forecast_result

def extract_metric_from_question(question: str) -> str:
    """
    Extract the metric from a forecasting question.
    
    Args:
        question: The forecasting question
        
    Returns:
        The metric name to use for forecasting
    """
    question = question.lower()
    
    if "net profit" in question:
        return "net_profit"
    elif "revenue" in question:
        return "revenue"
    elif "eps" in question or "earning" in question:
        return "basic_eps"
    else:
        return "net_income"  # Default metric

def run_forecast_from_question(company: str, question: str, periods: int = 4) -> Dict[str, Any]:
    """
    Run a forecast based on a forecasting question.
    
    Args:
        company: Company ticker symbol
        question: Forecasting question
        periods: Number of periods to forecast
        
    Returns:
        Dictionary with forecast results
    """
    # Extract the metric from the question
    metric = extract_metric_from_question(question)
    
    # Run the ARIMA forecast
    return arima_forecast(company, metric, periods=periods)

# Create tools for each forecasting method
time_series_tool = FunctionTool.from_defaults(
    fn=time_series_forecast,
    name="time_series_forecast",
    description="Forecast financial metrics using trend analysis"
)

moving_average_tool = FunctionTool.from_defaults(
    fn=moving_average_forecast,
    name="moving_average_forecast",
    description="Forecast financial metrics using simple moving average"
)

exp_smoothing_tool = FunctionTool.from_defaults(
    fn=exponential_smoothing_forecast,
    name="exponential_smoothing_forecast",
    description="Forecast financial metrics using exponential smoothing"
)

arima_tool = FunctionTool.from_defaults(
    fn=arima_forecast,
    name="arima_forecast",
    description="Forecast financial metrics using ARIMA model"
)

# Create the OpenAI agent with all the forecasting tools
agent = OpenAIAgent.from_tools(
    tools=[time_series_tool, moving_average_tool, exp_smoothing_tool, arima_tool],
    verbose=True,
    llm=llm,
    system_prompt="""
    You are a financial analysis assistant specializing in Colombo Stock Exchange companies.
    
    You have access to the following forecasting tools:
    1. time_series_forecast - Simple trend-based forecasting
    2. moving_average_forecast - Forecasting based on the moving average of recent periods
    3. exponential_smoothing_forecast - Gives more weight to recent observations
    4. arima_forecast - ARIMA model for time series forecasting
    
    When a user asks for financial forecasts, consider:
    - Which company they're interested in (REXP.N0000 or DIPD.N0000)
    - Which metric (revenue or net_income)
    - Which forecasting method would be most appropriate
    - How many periods to forecast (typically 1-4 quarters)
    
    Focus on providing accurate, concise, and actionable financial insights.
    """,
    temperature=0.2
)

def forecast_agent(query: str) -> str:
    """
    Function to query the OpenAI agent
    
    Args:
        query: User query about financial data
    
    Returns:
        Response from the agent
    """
    response = agent.chat(query)
    return str(response) 



print(forecast_agent("What is the forecast for REXP.N0000 revenue for the next 4 quarters?"))