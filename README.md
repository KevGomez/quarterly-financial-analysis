# Quarterly Financial Analysis

A tool for parsing, analyzing, and forecasting quarterly financial reports.

## Overview

This project automates the extraction of financial metrics from quarterly reports (PDFs), validates the data, and provides forecasting capabilities using time series analysis.

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
├── scrapper.py             # Scrapes additional data if needed
├── prompts.py              # Contains prompts for LLM interactions
├── query_engine.py         # Manages querying and data retrieval
├── reports_metadata.json   # Metadata for processed reports
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── README.md               # Project documentation
```

## Features

- PDF parsing and text extraction
- Financial metrics extraction using LLMs
- Data validation and consistency checks
- Time series forecasting with ARIMA and Exponential Smoothing
- Report organization and metadata management

## File Descriptions

- **parser.py**: Handles the extraction of text from PDF files, organizes PDF files by year and quarter, and processes reports from the UI. It also updates the metadata for processed reports.
- **forecaster_agent.py**: Manages the forecasting logic, including time series analysis and agent interactions for financial predictions.
- **app.py**: Contains the main application logic, including the Streamlit UI for displaying datasets, visualizing financial metrics, and managing user interactions.
- **scrapper.py**: Scrapes additional data if needed, though its specific functionality is not detailed in the current context.
- **prompts.py**: Contains prompts used for interactions with language models, facilitating natural language processing tasks.
- **query_engine.py**: Manages querying and data retrieval, including loading financial data from JSON files and handling query engine components.
- **reports_metadata.json**: Stores metadata for processed reports, including available years, quarters, and file mappings.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
4. Initialize the environment:
   ```bash
   python app.py --init
   ```

## Usage

Process all company reports:

```bash
python app.py --process
```

Process reports for a specific company:

```bash
python app.py --process --company AAPL
```

## Dependencies

- pandas, numpy, matplotlib: Data manipulation and visualization
- PyPDF2: PDF parsing
- llama-index: LLM integration
- statsmodels, scikit-learn: Time series forecasting
- streamlit: UI components (future development)
- Other utilities: tqdm, pyyaml, python-dotenv

## Troubleshooting

- **No datasets available**: Use the "Download Reports" and "Process Reports" buttons to collect data
- **PDF parsing errors**: Check your LLAMA_CLOUD_API_KEY in the .env file
- **Agent initialization failed**: Ensure that dataset folders contain processed text files
- **UI errors**: Check the console for error messages and ensure all required packages are installed
