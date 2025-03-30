# Financial Analysis Assistant

A comprehensive tool for financial analysis of companies listed on the Colombo Stock Exchange, with a focus on Richard Pieris PLC (REXP.N0000) and Dipped Products PLC (DIPD.N0000).

## Application Overview

This application provides an interactive interface for financial analysts to examine company performance through PDF report parsing, data visualization, question answering, and forecasting. It leverages LLM-powered agents to extract and analyze financial information from quarterly reports.

## Application Flow

1. **Data Collection**: Scrape quarterly financial reports as PDFs using `scraper.py`
2. **Data Processing**: Parse PDF reports into text files using `parser.py`
3. **Agent Setup**: Create query engines and agents in `query_engine.py`
4. **User Interface**: Interact with the data through Streamlit interface in `app.py`

## Detailed Workflow Descriptions

### Ask Questions Flow

When users ask questions about financial data through the UI:

1. **User Input**: 
   - User enters a query in the text input field
   - User selects one or more companies from the sidebar

2. **Agent Processing**:
   - The app checks `st.session_state.agent_initialized` to ensure agents are available
   - The query is combined with company context: `company_context + query`
   - The `get_agent_response()` function is called with this enhanced query
   - The main agent (`agent` from query_engine.py) processes the query using `agent.chat()`
   - The agent decides which tool to use based on the query content:
     - `rexp_tool`: For Richard Pieris-specific questions
     - `dipd_tool`: For Dipped Products-specific questions
     - `call_sub_agent_tool`: For complex multi-company queries

3. **Tool Execution**:
   - Selected tool queries the appropriate VectorStoreIndex 
   - Index retrieves relevant information from processed text files
   - Response is synthesized using the "compact" response mode

4. **Response Handling**:
   - The response is displayed in the UI
   - Question and answer are added to query history for future reference
   - The UI displays the answer with proper formatting

5. **Fallback Mechanism**:
   - If datasets aren't available, a `SimpleFallbackQueryEngine` returns a user-friendly message
   - If an error occurs, it's caught and displayed in the UI

### Visualization Flow

When users want to visualize financial metrics:

1. **User Selection**:
   - User selects a company from dropdown
   - User chooses a financial metric (Revenue, Profit, EPS, Operating Expenses)
   - User can opt to compare multiple companies

2. **Data Extraction**:
   - `get_financial_quarters()` function is called to retrieve quarter identifiers:
     - Creates a specific prompt for extraction
     - Queries the appropriate tool (rexp_tool or dipd_tool)
     - Parses the response to extract quarter labels
     - Falls back to sample data if extraction fails
   
   - `get_financial_metric()` function is called to retrieve metric values:
     - Creates a specific prompt for the chosen metric
     - Queries the appropriate company-specific tool
     - Parses the response to extract numerical values
     - Falls back to sample data if extraction fails

3. **Visualization Generation**:
   - Data is compiled into DataFrames
   - Altair charts are generated based on the extracted data
   - Charts are displayed in the UI with proper formatting
   - Raw data tables are also displayed for reference

4. **Error Handling**:
   - Multiple try-except blocks ensure graceful failure
   - Fallback to sample data if agent queries fail
   - Clear error messages guide the user when issues occur
   - Empty dataset warnings when no data is available

### Forecasting Flow

When users request financial forecasts:

1. **User Configuration**:
   - User selects a company from dropdown
   - User chooses a financial metric to forecast
   - User specifies the number of quarters to forecast (1-8)
   - User initiates the forecast with the "Generate Forecast" button

2. **Historical Data Retrieval**:
   - `get_financial_quarters()` and `get_financial_metric()` functions retrieve historical data
   - The same agent querying process occurs as in the Visualization flow
   - Data validation ensures sufficient history is available (minimum 4 quarters)

3. **Forecast Calculation**:
   - Base value is identified from the last historical data point
   - Average growth is calculated from historical trends
   - Future values are projected by applying average growth to the base
   - Future quarter labels are generated by extending from the last historical quarter

4. **Result Presentation**:
   - Historical and forecast data are combined in a unified DataFrame
   - An Altair chart visualizes both historical and forecast values with different colors
   - Summary metrics are displayed (current value, forecast value, growth percentages)
   - Detailed forecast data is available in an expandable table

5. **Error Handling**:
   - Validation ensures enough historical data exists for meaningful forecasting
   - Try-except blocks capture and display errors
   - Fallback to sample data when agent queries fail

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-analysis-assistant.git
cd financial-analysis-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys
```

## Project Structure

```
financial-analysis-assistant/
├── app.py                  # Streamlit UI application
├── parser.py               # PDF parsing functionality
├── query_engine.py         # Agent and query engine setup
├── scraper.py              # PDF report scraper
├── Company1_Reports/       # PDF reports for Richard Pieris
├── Company2_Reports/       # PDF reports for Dipped Products
├── Company1_Datasets/      # Parsed text for Richard Pieris
├── Company2_Datasets/      # Parsed text for Dipped Products
└── README.md               # Documentation
```

## File Documentation

### scraper.py

The `scraper.py` file is responsible for downloading quarterly financial reports from company websites:

```python
def download_company_reports(company_code, start_date, end_date):
    """
    Downloads quarterly reports for a specific company within a date range.
    
    Args:
        company_code (str): The code for the company (e.g., "REXP.N0000")
        start_date (datetime): The start date to fetch reports from
        end_date (datetime): The end date to fetch reports until
        
    Returns:
        int: The number of reports downloaded
    """
    # Map company code to folder and URL patterns
    company_folder = COMPANY_FOLDERS.get(company_code)
    
    # Create folders if they don't exist
    os.makedirs(company_folder, exist_ok=True)
    
    # Logic to fetch reports from company websites or CSE
    # ...
```

### parser.py

The `parser.py` file handles the conversion of PDF reports to text:

```python
def process_company_reports(company_folder, dataset_folder):
    """
    Process all PDF reports for a company
    
    Args:
        company_folder (str): Folder containing PDF reports
        dataset_folder (str): Folder to save parsed text files
    """
    # Set up LlamaParse for PDF extraction
    parser = LlamaParse(
        result_type="text",
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY")
    )
    
    # Configure file extractor for PDFs
    file_extractor = {".pdf": parser}
    
    # Use SimpleDirectoryReader to parse all PDFs
    documents = SimpleDirectoryReader(
        input_dir=company_folder,
        file_extractor=file_extractor
    ).load_data()
    
    # Save each document as text
    for i, doc in enumerate(documents):
        output_file = os.path.join(dataset_folder, f"parsed_doc_{i+1}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(doc.text)
```

### query_engine.py

The `query_engine.py` file sets up the query engines and agents:

```python
def create_query_engine(folder_path):
    """
    Create a query engine for a specific company's documents
    
    Args:
        folder_path (str): Path to the folder containing parsed documents
        
    Returns:
        QueryEngine: A query engine for the documents
    """
    # Check if folder exists and has files
    if not os.path.exists(folder_path) or not [f for f in os.listdir(folder_path) if f.endswith('.txt')]:
        return create_fallback_engine(folder_path)
    
    # Load documents from the dataset folder
    documents = SimpleDirectoryReader(input_dir=folder_path).load_data()
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(documents)
    
    # Create query engine with improved response synthesis
    return index.as_query_engine(response_mode="compact", verbose=True)
```

The file also creates two specialized agents:

1. **Sub-agent**: Handles company-specific queries with a focus on accurate financial metrics extraction
2. **Main agent**: Coordinates all tools and provides financial analysis with a broader context

### app.py

The `app.py` file implements the Streamlit interface:

```python
def main():
    """Main function to set up the Streamlit UI"""
    # Set up app configuration
    st.set_page_config(
        page_title="Financial Analysis Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Display title
    st.title("Financial Analysis Assistant")
    
    # Set up sidebar for company selection and date range
    with st.sidebar:
        st.header("Settings")
        
        # Company selection 
        companies = ["REXP.N0000", "DIPD.N0000"]
        selected = st.multiselect(
            "Select Companies",
            options=companies,
            default=companies[0],
            format_func=lambda x: COMPANY_NAMES.get(x, x)
        )
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                get_default_start_date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now()
            )
        
        # Buttons for downloading and processing
        if st.button("Download Reports"):
            # Download logic
            pass
            
        if st.button("Process Reports"):
            # Processing logic
            pass
            
        # Display dataset information
        display_dataset_info()
    
    # Main window with tabs
    main_tabs = st.tabs(["Datasets", "Insights"])
    
    # Datasets tab - display files and content
    with main_tabs[0]:
        # File viewer interface
        pass
    
    # Insights tab - Q&A and visualizations
    with main_tabs[1]:
        insight_tabs = st.tabs(["Ask Questions", "Visualize Data", "Compare Companies", "Forecast"])
        
        # Tab implementations for each analysis mode
        # ...
```

## Key Features

1. **Automated Data Collection**: Download quarterly financial reports from company websites
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