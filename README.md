# Quarterly Financial Analysis Dashboard

A comprehensive tool for parsing, analyzing, and forecasting quarterly financial reports for Sri Lankan publicly listed companies.

## Overview

This project automates the extraction of financial metrics from quarterly reports (PDFs), validates the data, builds financial datasets, and provides forecasting capabilities using time series analysis algorithms. The system leverages language models for intelligent data extraction and presents results through an interactive Streamlit dashboard.

## Project Structure

```
.
├── REXP_Datasets/          # Contains datasets for Richard Pieris PLC
├── REXP_Reports/           # Contains raw reports for Richard Pieris PLC
├── DIPD_Datasets/          # Contains datasets for Dipped Products PLC
├── DIPD_Reports/           # Contains raw reports for Dipped Products PLC
├── parser.py               # Handles PDF parsing and data extraction
├── forecaster_agent.py     # Manages forecasting logic and agent interactions
├── app.py                  # Main application logic and Streamlit UI
├── scrapper.py             # Scrapes quarterly reports from CSE website
├── prompts.py              # Contains prompts for LLM interactions
├── query_engine.py         # Manages querying and data retrieval
├── reports_metadata.json   # Metadata for processed reports
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── README.md               # Project documentation
```

## Features

- Automated scraping of quarterly financial reports from the Colombo Stock Exchange website
- PDF parsing and text extraction with company-specific processing logic
- Financial metrics extraction using language models
- Data validation and consistency checks
- Metadata tracking for reports and financial metrics
- Time series forecasting with multiple algorithms:
  - ARIMA (Auto-Regressive Integrated Moving Average)
  - Exponential Smoothing
  - Moving Average
- Interactive data visualization and comparison tools
- Natural language query interface for financial data
- Report organization and metadata management

## Core Mechanisms

### 1. Data Collection Pipeline

The system uses `scrapper.py` to automatically fetch quarterly financial reports from the Colombo Stock Exchange website:

- Configurable to target specific companies and date ranges
- Parallel download capabilities for faster retrieval
- Intelligent filtering to identify quarterly reports
- Organization of reports by company and quarter

### 2. Financial Data Extraction

`parser.py` implements a sophisticated extraction pipeline:

- Extracts text from PDF files with company-specific processing logic
- Classifies reports by year and quarter using pattern matching and LLM verification
- Extracts structured financial metrics using prompt engineering and LLMs
- Validates data consistency and completeness
- Organizes data in standardized JSON format
- Maintains comprehensive metadata for processing history

### 3. Data Modeling and Forecasting

`forecaster_agent.py` provides multiple forecasting methods:

- Automatic selection of optimal ARIMA parameters based on data characteristics
- Exponential smoothing for trend and seasonality modeling
- Moving average for baseline forecasting
- Uncertainty quantification with prediction intervals
- Model evaluation and comparison metrics

### 4. Query and Retrieval System

`query_engine.py` enables natural language interaction with financial data:

- Structured data models for quarterly financial information
- Query engine for retrieving specific financial metrics
- Sub-agent architecture for processing complex financial queries
- Schema definition for financial metrics consistency

### 5. Interactive Dashboard

`app.py` implements a Streamlit-based user interface with:

- Workflow-based navigation system
- Dynamic data visualization with interactive charts
- Metric comparison across companies and time periods
- Financial forecasting with configurable parameters
- Natural language query capabilities
- Report management and processing controls

## File Descriptions

- **parser.py**: Handles the extraction of text from PDF files with company-specific processing rules, organizes PDF files by year and quarter, and processes reports. It includes mechanisms for LLM-based extraction of financial metrics and maintains metadata about processed reports.

- **forecaster_agent.py**: Implements multiple time series forecasting algorithms including ARIMA, Exponential Smoothing, and Moving Average. It contains logic for optimal model parameter selection, visualization of forecasts, and uncertainty quantification.

- **app.py**: Contains the main application logic built with Streamlit, providing an interactive dashboard with workflow-based navigation. It includes visualization components, report processing controls, and integrates all other modules.

- **scrapper.py**: Implements web scraping capabilities to download quarterly financial reports from the Colombo Stock Exchange website with intelligent filtering to identify reports by quarter and year.

- **prompts.py**: Contains structured prompts for language model interactions, focusing on financial data extraction from quarterly reports with specific formatting requirements.

- **query_engine.py**: Manages structured data models for financial information and implements query mechanisms for retrieving specific metrics. It includes schema definitions and tools for natural language querying.

- **reports_metadata.json**: Stores metadata for processed reports, including available years, quarters, file mappings, and processing status.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here  # Optional for enhanced PDF parsing
   ```
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Usage

The application follows a workflow-based approach:

1. **Download Reports**: Fetch quarterly reports from the CSE website
2. **Process Reports**: Extract financial data from the downloaded PDFs
3. **Data Visualization**: Explore financial metrics across quarters and companies
4. **Financial Forecast**: Generate forecasts for selected metrics

## Technologies Used

- **PDF Parsing**: PyPDF2 for text extraction
- **NLP/LLM**: OpenAI APIs (GPT models) for financial data extraction
- **Data Processing**: Pandas and NumPy for data manipulation
- **Time Series Forecasting**: Statsmodels (ARIMA, Exponential Smoothing)
- **Machine Learning**: Scikit-learn for additional modeling
- **Web Scraping**: Requests, BeautifulSoup, and Selenium
- **UI**: Streamlit for the interactive dashboard
- **Visualization**: Matplotlib and Altair

## Troubleshooting

- **No datasets available**: Use the "Download Reports" and "Process Reports" buttons to collect data
- **PDF parsing errors**: Check your LLAMA_CLOUD_API_KEY in the .env file
- **Agent initialization failed**: Ensure that dataset folders contain processed text files
- **UI errors**: Check the console for error messages and ensure all required packages are installed

## Advanced Configuration

The system includes several customization options:

- Configure additional companies by updating the COMPANY_FOLDERS dictionary
- Adjust LLM prompts in prompts.py for different financial report formats
- Modify forecasting parameters in forecaster_agent.py for different prediction horizons
