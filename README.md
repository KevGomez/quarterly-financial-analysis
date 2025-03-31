# Quarterly Financial Analysis

A tool for parsing, analyzing, and forecasting quarterly financial reports.

## Overview

This project automates the extraction of financial metrics from quarterly reports (PDFs), validates the data, and provides forecasting capabilities using time series analysis.

## Project Structure

```
.
├── src/
│   ├── agents/             # Agent-related functionality
│   ├── ingestion/          # PDF parsing and data ingestion
│   ├── metrics/            # Financial metrics extraction and validation
│   └── utils/              # Utility functions and helpers
├── data/                   # Storage for reports and processed data
├── config.yaml             # Configuration settings
├── main.py                 # Application entry point
└── requirements.txt        # Dependencies
```

## Features

- PDF parsing and text extraction
- Financial metrics extraction using LLMs
- Data validation and consistency checks
- Time series forecasting with ARIMA and Exponential Smoothing
- Report organization and metadata management

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Initialize the environment:
   ```
   python main.py --init
   ```

## Usage

Process all company reports:

```
python main.py --process
```

Process reports for a specific company:

```
python main.py --process --company AAPL
```

## Dependencies

- pandas, numpy, matplotlib: Data manipulation and visualization
- PyPDF2: PDF parsing
- llama-index: LLM integration
- statsmodels, scikit-learn: Time series forecasting
- streamlit: UI components (future development)
- Other utilities: tqdm, pyyaml, python-dotenv

2. **PDF Processing**: Convert PDF reports to searchable text using LlamaParse
3. **Question Answering**: Query financial data using natural language
4. **Visualization**: Generate charts for key financial metrics
5. **Comparison**: Compare financial performance across companies
6. **Forecasting**: Project future financial performance based on historical trends

## Troubleshooting

- **No datasets available**: Use the "Download Reports" and "Process Reports" buttons to collect data
- **PDF parsing errors**: Check your LLAMA_CLOUD_API_KEY in the .env file
- **Agent initialization failed**: Ensure that dataset folders contain processed text files
- **UI errors**: Check the console for error messages and ensure all required packages are installed
