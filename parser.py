import os
import json
from datetime import datetime
import re
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from prompts import single_report_prompt
from query_engine import quarterly_financial_data_tool
from PyPDF2 import PdfReader
from typing import Any
# Load environment variables
load_dotenv()

# Constants for folder structure
COMPANY_FOLDERS = {
    "REXP.N0000": "REXP_Reports",
    "DIPD.N0000": "DIPD_Reports"
}

# Create dataset folders for each company
DATASET_FOLDERS = {
    "REXP.N0000": "REXP_Datasets",
    "DIPD.N0000": "DIPD_Datasets"
}


# Create dataset folders if they don't exist
for folder in DATASET_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)
    # Create audit logs directory (for tracking extraction process)
    os.makedirs(os.path.join(folder, "audit_logs"), exist_ok=True)

def ensure_metadata_file():
    """Ensure that the reports_metadata.json file exists and has a valid structure"""
    metadata_file = "reports_metadata.json"
    default_metadata = {}
    
    # Initialize with default structure for each company
    for company_code in COMPANY_FOLDERS:
        default_metadata[company_code] = {
            "available_years": [],
            "quarters_by_year": {},
            "missing_quarters": {},
            "raw_files": {},
            "file_mapping": {},
            "fiscal_calendar": {
                "Q1": "Jun",
                "Q2": "Sep",
                "Q3": "Dec",
                "Q4": "Mar"
            },
            "last_updated": datetime.now().isoformat()
        }
    
    # If file doesn't exist or is malformed, create it with default structure
    need_init = False
    if not os.path.exists(metadata_file):
        need_init = True
    else:
        # Try to load to verify format
        try:
            with open(metadata_file, 'r') as f:
                json.load(f)
        except Exception as e:
            print(f"Metadata file exists but is malformed: {e}. Reinitializing.")
            need_init = True
    
    if need_init:
        try:
            with open(metadata_file, 'w') as f:
                json.dump(default_metadata, f, indent=2)
            print(f"Initialized reports_metadata.json with default structure")
        except Exception as e:
            print(f"Error initializing metadata file: {e}")

# Ensure metadata file exists and has valid structure
ensure_metadata_file()

def create_query_engine(company_folder):
    """Create a query engine for the given company folder"""
    if not os.path.exists(company_folder):
        print(f"Folder not found: {company_folder}")
        return None
    
    # Get list of PDF files in the folder
    pdf_files = [os.path.join(company_folder, f) for f in os.listdir(company_folder) 
                if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {company_folder}")
        return None
    
    print(f"Loading {len(pdf_files)} PDF files from {company_folder}")
    
    try:
        # Load the documents
        documents = SimpleDirectoryReader(
            input_files=pdf_files
        ).load_data()
        
        # Create an index from the documents
        index = VectorStoreIndex.from_documents(documents)
        
        # Return a query engine
        return index.as_query_engine(similarity_top_k=5)
    
    except Exception as e:
        print(f"Error creating query engine for {company_folder}: {str(e)}")
        return None

def create_query_engine_for_single_file(pdf_file_path):
    """Create a query engine for a single PDF file"""
    try:
        # Determine the company code from the filename
        company_code = None
        if '770_' in pdf_file_path or '771_' in pdf_file_path:
            company_code = "REXP.N0000"
        elif '670_' in pdf_file_path:
            company_code = "DIPD.N0000"
            
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file_path, company_code=company_code)
        if not text:
            print(f"No text could be extracted from {pdf_file_path}")
            return None
            
        # Create a more structured document with metadata
        doc = Document(
            text=text,
            metadata={
                'source': os.path.basename(pdf_file_path),
                'type': 'financial_report',
                'company_code': company_code,
                'extracted_date': datetime.now().isoformat()
            },
            excluded_llm_metadata_keys=['extracted_date']
        )
        
        # Create index from the document with specific settings for financial data
        index = VectorStoreIndex.from_documents(
            [doc],
            service_context=None  # Use default
        )
        
        # Create query engine with specific response synthesis for financial data
        query_engine = index.as_query_engine(
            response_mode="compact",
            verbose=True,
            # Structured output enforcer
            output_formatter=lambda x: x.strip().replace('\n', ' ').replace('  ', ' ')
        )
        
        return query_engine
        
    except Exception as e:
        print(f"Error creating query engine for {pdf_file_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path, max_pages=8, company_code=None):
    """Extract text from PDF file with improved structure preservation for financial tables"""
    try:
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return None
            
        # Open the PDF
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf = PdfReader(file)
            
            # Get total number of pages
            num_pages = len(pdf.pages)
            
            # Limit pages to process
            pages_to_read = min(max_pages, num_pages)
            
            # Initialize text storage
            text_content = []
            
            # Company-specific processing logic
            if company_code == "DIPD.N0000":
                # DIPD reports typically have the quarterly financials in the first few pages
                # Look specifically for the Company Statement section
                for page_num in range(min(pages_to_read, 5)):  # Focus on first 5 pages for DIPD
                    try:
                        # Get the page
                        page = pdf.pages[page_num]
                        
                        # Extract text with table structure preservation
                        text = page.extract_text()
                        
                        # Clean and normalize the text
                        if text:
                            # Add DIPD-specific markers for better extraction
                            if "Revenue from contracts with customers" in text:
                                text = text.replace("Revenue from contracts with customers", 
                                                  "### DIPD REVENUE: Revenue from contracts with customers ###")
                            
                            # Mark Group and Company columns
                            if "Group" in text and "Company" in text:
                                text = text.replace("Group", "### GROUP COLUMN ###").replace("Company", "### COMPANY COLUMN ###")
                            
                            lines = text.split('\n')
                            processed_lines = []
                            
                            for line in lines:
                                # Clean up each line
                                clean_line = ' '.join(line.split())
                                
                                # Process DIPD-specific line items with more precise markers
                                for marker in [
                                    ("Revenue from contracts with customers", "REVENUE"),
                                    ("Cost of Sales", "COGS"),
                                    ("Gross Profit", "GROSS_PROFIT"),
                                    ("Distribution Costs", "DISTRIBUTION_COSTS"),
                                    ("Administrative Expenses", "ADMIN_EXPENSES"),
                                    ("Finance Income", "FINANCE_INCOME"),
                                    ("Finance Costs", "FINANCE_COSTS"),
                                    ("Profit Before Tax", "PROFIT_BEFORE_TAX"),
                                    ("Tax Expense", "TAX_EXPENSE"),
                                    ("Profit for the Period", "NET_PROFIT"),
                                    ("Basic Earnings Per Share", "EPS")
                                ]:
                                    if marker[0] in clean_line:
                                        clean_line = f"## DIPD METRIC: {marker[1]} | {clean_line} ##"
                                
                                # Add special markers for period identification
                                for period_marker in [
                                    "Three months ended",
                                    "Six months ended",
                                    "Nine months ended",
                                    "Twelve months ended"
                                ]:
                                    if period_marker.lower() in clean_line.lower():
                                        clean_line = f"### PERIOD: {clean_line} ###"
                                
                                processed_lines.append(clean_line)
                            
                            # Preserve line breaks for table structure
                            processed_text = '\n'.join(processed_lines)
                            text_content.append(processed_text)
                    
                    except Exception as e:
                        print(f"Error extracting text from DIPD page {page_num + 1}: {e}")
                        continue
            
            elif company_code == "REXP.N0000":
                # REXP reports have a different structure
                for page_num in range(pages_to_read):
                    try:
                        # Get the page
                        page = pdf.pages[page_num]
                        
                        # Extract text with table structure preservation
                        text = page.extract_text()
                        
                        # Clean and normalize the text
                        if text:
                            # Add REXP-specific markers for better extraction
                            if "Revenue" in text or "Turnover" in text:
                                text = text.replace("Revenue", "### REXP REVENUE: Revenue ###")
                                text = text.replace("Turnover", "### REXP REVENUE: Turnover ###")
                            
                            if "Group" in text and "Company" in text:
                                text = text.replace("Group", "### GROUP ###").replace("Company", "### COMPANY ###")
                            
                            lines = text.split('\n')
                            processed_lines = []
                            
                            for line in lines:
                                # Clean up each line
                                clean_line = ' '.join(line.split())
                                
                                # Process REXP-specific line items
                                for marker in ["Revenue", "Turnover", "Cost of Sales", "Gross Profit", 
                                             "Distribution Costs", "Administrative Expenses",
                                             "Results from Operating Activities", "Operating Profit",
                                             "Finance Income", "Finance Costs", "Profit Before Income Tax",
                                             "Income Tax Expense", "Profit for the Period", "Earnings per Share"]:
                                    if marker in clean_line:
                                        clean_line = f"## REXP METRIC: {clean_line} ##"
                                
                                processed_lines.append(clean_line)
                            
                            # Preserve line breaks for table structure
                            processed_text = '\n'.join(processed_lines)
                            text_content.append(processed_text)
                    
                    except Exception as e:
                        print(f"Error extracting text from REXP page {page_num + 1}: {e}")
                        continue
            
            else:
                # Generic processing for other companies
                # Process each page
                for page_num in range(pages_to_read):
                    try:
                        # Get the page
                        page = pdf.pages[page_num]
                        
                        # Extract text with table structure preservation
                        text = page.extract_text()
                        
                        # Clean and normalize the text
                        if text:
                            # Keep line breaks for table structure
                            lines = text.split('\n')
                            processed_lines = []
                            
                            for line in lines:
                                # Clean up each line
                                clean_line = ' '.join(line.split())
                                
                                # Add special markers for financial tables
                                for marker in ["Three months ended", "Quarter ended", "3 months ended", 
                                             "Six months ended", "Nine months ended", "Twelve months ended"]:
                                    if marker.lower() in clean_line.lower():
                                        clean_line = f"### {clean_line} ###"
                                
                                # Add special markers for financial statement sections
                                for marker in ["Statement of Profit or Loss", "Income Statement", 
                                             "Consolidated Statement", "Revenue", "REVENUE"]:
                                    if marker in clean_line:
                                        clean_line = f"## {clean_line} ##"
                                
                                processed_lines.append(clean_line)
                            
                            # Preserve line breaks for table structure
                            processed_text = '\n'.join(processed_lines)
                            text_content.append(processed_text)
                            
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            # Combine all text with proper spacing
            combined_text = '\n\n'.join(text_content)
            
            # Add structural markers for financial statements
            financial_markers = [
                "Statement of Profit or Loss",
                "Statement of Comprehensive Income",
                "Statement of Financial Position",
                "Income Statement",
                "Balance Sheet",
                "Cash Flow Statement",
                "Consolidated Statement"
            ]
            
            # Add section markers for better context
            for marker in financial_markers:
                combined_text = combined_text.replace(marker, f"\n=== {marker} ===\n")
            
            # Add markers for quarter periods
            period_markers = [
                "Three months ended",
                "3 months ended", 
                "6 months ended",
                "Six months ended",
                "9 months ended",
                "Nine months ended",
                "12 months ended",
                "Twelve months ended",
                "Quarter ended"
            ]
            
            for marker in period_markers:
                pattern = re.compile(f"({marker}.*?)(\\d{{1,2}}(?:st|nd|rd|th)?\\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+20\\d{{2}})", re.IGNORECASE)
                combined_text = pattern.sub(r"\n*** PERIOD: \1\2 ***\n", combined_text)
            
            # Add specific markers for company type (if available)
            if company_code:
                combined_text = f"COMPANY CODE: {company_code}\n\n" + combined_text
            
            return combined_text if combined_text.strip() else None
            
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

def get_pdf_files_by_year(company_folder):
    """Organize PDF files by detected year and quarter using regex and LLM verification."""
    if not os.path.exists(company_folder):
        return {}

    files_by_year = {}
    all_files = [f for f in os.listdir(company_folder) if f.lower().endswith('.pdf')]

    for file in all_files:
        file_path = os.path.join(company_folder, file)
        
        # Extract quarter and year using regex patterns and LLM verification
        quarter, year = extract_quarter_from_pdf_content(file_path)
        
        # If year not determined, try basic pattern from filename or use file creation time
        if not year:
            year_match = re.search(r'20\d{2}', file)
            if year_match:
                year = year_match.group(0)
            else:
                year = str(datetime.fromtimestamp(os.path.getctime(file_path)).year)

        # Save results
        if year not in files_by_year:
            files_by_year[year] = []
        files_by_year[year].append({
            'path': file_path,
            'name': file,
            'quarter': quarter
        })

        print(f"File: {file} | Year: {year} | Quarter: {quarter if quarter else '❌ Not Detected'}")

    return files_by_year



def process_reports_from_ui(company_codes, start_year=None, end_year=None, progress_callback=None, selected_quarter=None):
    """
    Process reports for specified companies and date range from the UI.
    
    Args:
        company_codes: List of company codes to process
        start_year: Start year for processing (int)
        end_year: End year for processing (int)
        progress_callback: Optional callback function to report progress
        selected_quarter: Optional specific quarter to process (e.g., "Q1", "Q2", etc.)
        
    Returns:
        Dictionary with results of the processing operation
    """
    if not isinstance(company_codes, list):
        company_codes = [company_codes]
    
    results = {}
    processed_data = {}  # Data for updating metadata
    
    # Load existing metadata
    try:
        with open("reports_metadata.json", 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        metadata = {}
    
    total_files_to_process = 0
    files_to_process = []
    
    # First, collect all files that need to be processed
    for company_code in company_codes:
        company_folder = COMPANY_FOLDERS.get(company_code)
        if not company_folder or company_code not in metadata:
            results[company_code] = {"status": "error", "message": f"Unknown company code or missing metadata: {company_code}"}
            continue
        
        # Initialize processed data for this company
        processed_data[company_code] = {
            "years_processed": [],
            "quarters_processed": {}
        }
        
        # Get the raw files from metadata
        raw_files = metadata[company_code].get("raw_files", {})
        
        # Determine which years to process
        years_to_process = []
        if start_year is not None and end_year is not None:
            years_to_process = [str(year) for year in range(start_year, end_year + 1)]
        else:
            years_to_process = list(raw_files.keys())
            years_to_process.remove("unknown") if "unknown" in years_to_process else None
        
        # For each year
        for year in years_to_process:
            if year not in raw_files:
                continue
                
            # Get quarters for this year
            quarters = raw_files[year]
            
            # Filter by selected quarter if specified
            if selected_quarter:
                if selected_quarter in quarters:
                    quarter_files = quarters[selected_quarter]
                    for pdf_file in quarter_files:
                        files_to_process.append({
                            "company_code": company_code,
                            "year": year,
                            "quarter": selected_quarter,
                            "file": pdf_file,
                            "path": os.path.join(company_folder, pdf_file)
                        })
                    total_files_to_process += len(quarter_files)
            else:
                # Process all quarters
                for quarter, quarter_files in quarters.items():
                    for pdf_file in quarter_files:
                        files_to_process.append({
                            "company_code": company_code,
                            "year": year,
                            "quarter": quarter,
                            "file": pdf_file,
                            "path": os.path.join(company_folder, pdf_file)
                        })
                    total_files_to_process += len(quarter_files)
    
    # Now process each file
    for idx, file_info in enumerate(files_to_process):
        company_code = file_info["company_code"]
        year = file_info["year"]
        quarter = file_info["quarter"]
        pdf_path = file_info["path"]
        
        # Update progress if callback provided
        if progress_callback:
            progress = (idx + 1) / total_files_to_process
            progress_callback(progress)
        
        try:
            # Create query engine for this file
            query_engine = create_query_engine_for_single_file(pdf_path)
            if not query_engine:
                results[f"{company_code}_{year}_{quarter}"] = {
                    "status": "error",
                    "message": f"Could not create query engine for {pdf_path}"
                }
                continue
            
            # Extract financial metrics using the query engine
            metrics = extract_financial_metrics(query_engine, company_code)
            
            if metrics:
                # Add quarter and year information
                metrics["quarter"] = quarter
                metrics["year"] = year
                
                # Save metrics to dataset folder
                dataset_folder = DATASET_FOLDERS.get(company_code)
                if dataset_folder:
                    os.makedirs(dataset_folder, exist_ok=True)
                    
                    # Create filename based on quarter and year
                    metrics_file = f"{quarter}_{year}_metrics.json"
                    metrics_path = os.path.join(dataset_folder, metrics_file)
                    
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    # Update processed data
                    if year not in processed_data[company_code]["years_processed"]:
                        processed_data[company_code]["years_processed"].append(year)
                    
                    if year not in processed_data[company_code]["quarters_processed"]:
                        processed_data[company_code]["quarters_processed"][year] = []
                    
                    if quarter not in processed_data[company_code]["quarters_processed"][year]:
                        processed_data[company_code]["quarters_processed"][year].append(quarter)
                    
                    results[f"{company_code}_{year}_{quarter}"] = {
                        "status": "success",
                        "message": f"Successfully processed {quarter} {year}"
                    }
                else:
                    results[f"{company_code}_{year}_{quarter}"] = {
                        "status": "error",
                        "message": f"Dataset folder not found for {company_code}"
                    }
            else:
                results[f"{company_code}_{year}_{quarter}"] = {
                    "status": "error",
                    "message": f"Could not extract metrics from {pdf_path}"
                }
        
        except Exception as e:
            results[f"{company_code}_{year}_{quarter}"] = {
                "status": "error",
                "message": f"Error processing {pdf_path}: {str(e)}"
            }
    
    # Update metadata with processed data
    update_reports_metadata(processed_data)
    
    return results

def construct_extraction_query(company_code: str | None = None) -> str:
    """Construct the query for extracting financial metrics based on company code.
    
    Args:
        company_code: Optional company code to customize the query
        
    Returns:
        Query string for extracting financial metrics
    """
    if company_code == "DIPD.N0000":
        return """Please extract the following financial metrics from the Company column (not Group column) of the Income Statement:
        1. Revenue
        2. Cost of Sales (COGS)
        3. Gross Profit
        4. Distribution Costs
        5. Administrative Expenses
        6. Operating Income (calculated as Gross Profit - Operating Expenses)
        7. Finance Income
        8. Finance Costs
        9. Profit Before Tax
        10. Income Tax
        11. Net Profit
        12. Basic EPS
        
        Important rules:
        - Use ONLY the Company column data, not Group data
        - Operating Expenses = Distribution Costs + Administrative Expenses
        - Operating Income = Gross Profit - Operating Expenses
        - For Q3, use 9-month figures if available
        - All values should be in Rs.'000
        - Format the output as a JSON object with the following structure:
        {
            "revenue": 1234567,
            "cost_of_sales": 1234567,
            "gross_profit": 1234567,
            "distribution_costs": 1234567,
            "administrative_expenses": 1234567,
            "operating_income": 1234567,
            "finance_income": 1234567,
            "finance_costs": 1234567, 
            "profit_before_tax": 1234567,
            "income_tax": 1234567,
            "net_profit": 1234567,
            "basic_eps": 12.34
        }"""
    else:  # Default case including REXP.N0000
        return """Please extract the following financial metrics from the Consolidated Income Statement:
        1. Revenue/Turnover (in thousands)
        2. Cost of Goods Sold/Cost of Sales (in thousands) 
        3. Gross Profit (in thousands)
        4. Total Operating Expenses (in thousands)
        5. Operating Income/Operating Profit (in thousands)
        6. Finance Income (in thousands)
        7. Finance Cost (in thousands)
        8. Profit Before Tax (in thousands)
        9. Tax Expense (in thousands)
        10. Net Income/Profit after tax (in thousands)
        11. Earnings Per Share (EPS)
        
        Important rules:
        - Use ONLY the Group/Consolidated figures, NOT company-specific statements
        - Look for terms like "Group unaudited 03 months" column in the Income Statement
        - For Q3, use 3-month figures if available, not year-to-date
        - All monetary values should be in thousands (Rs.'000)
        - Format the output as a JSON object with the following structure:
        {
            "revenue": 1234567,
            "cost_of_sales": 1234567, 
            "gross_profit": 1234567,
            "operating_expenses": 1234567,
            "operating_income": 1234567,
            "finance_income": 1234567,
            "finance_costs": 1234567, 
            "profit_before_tax": 1234567,
            "income_tax": 1234567,
            "net_profit": 1234567,
            "basic_eps": 12.34
        }
        
        Return ONLY the JSON object, no additional text."""

def extract_financial_metrics(query_engine: Any, company_code: str | None = None) -> dict | None:
    """Extract financial metrics from a report using the query engine.
    
    Args:
        query_engine: The query engine to use for extraction
        company_code: Optional company code to customize extraction
        
    Returns:
        Dictionary of extracted metrics or None if extraction fails
    """
    if not query_engine:
        print("No query engine provided")
        return None
        
    try:
        # Call query instead of chat on the query_engine
        response = query_engine.query(construct_extraction_query(company_code))
        if not response or not hasattr(response, 'response'):
            print("No response from query engine")
            return None
            
        response_text = str(response.response)
        if not response_text:
            print("Empty response text")
            return None
            
        # Process JSON string - now always returns a string
        processed_json = preprocess_json_string(response_text)
        
        # Parse JSON
        try:
            metrics = json.loads(processed_json)
            if not isinstance(metrics, dict):
                print("Extracted metrics not in expected dictionary format")
                return None
                
            # Return empty dict if we got an empty JSON object
            if not metrics:
                print("Empty metrics object returned")
                return None
                
            # Define required fields based on company code
            if company_code == "DIPD.N0000":
                # DIPD has detailed breakdowns of expenses
                required_fields = [
                    "revenue", "cost_of_sales", "gross_profit",
                    "distribution_costs", "administrative_expenses",
                    "operating_income", "finance_income", "finance_costs",
                    "profit_before_tax", "income_tax", "net_profit",
                    "basic_eps"
                ]
            else:
                # REXP and other companies have operating_expenses instead of distribution/admin breakdown
                required_fields = [
                    "revenue", "cost_of_sales", "gross_profit",
                    "operating_expenses", "operating_income", 
                    "finance_income", "finance_costs",
                    "profit_before_tax", "income_tax", "net_profit",
                    "basic_eps"
                ]
            
            missing_fields = [field for field in required_fields if field not in metrics]
            if missing_fields:
                print(f"Missing required fields: {', '.join(missing_fields)}")
                # Check if we can fill in any missing fields with calculated values
                
                # Special handling for operating_income = gross_profit - operating_expenses
                if "operating_income" in missing_fields and "gross_profit" in metrics and "operating_expenses" in metrics:
                    try:
                        metrics["operating_income"] = metrics["gross_profit"] - metrics["operating_expenses"]
                        missing_fields.remove("operating_income")
                    except (TypeError, ValueError) as e:
                        print(f"Couldn't calculate operating_income: {e}")
                
                # Special handling for DIPD if operating_expenses is missing but we have components
                if company_code == "DIPD.N0000" and "operating_expenses" not in metrics and "distribution_costs" in metrics and "administrative_expenses" in metrics:
                    try:
                        metrics["operating_expenses"] = metrics["distribution_costs"] + metrics["administrative_expenses"]
                    except (TypeError, ValueError) as e:
                        print(f"Couldn't calculate operating_expenses: {e}")
                
                # If we still have missing fields, fail validation
                if missing_fields:
                    return None
                
            return metrics
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse metrics JSON: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error extracting metrics: {str(e)}")
        return None

def update_reports_metadata(processed_data):
    """
    Update the metadata file with information about processed reports.
    
    Args:
        processed_data: Dictionary containing information about processed reports
    """
    metadata_file = "reports_metadata.json"
    
    # Ensure the metadata file exists with valid structure
    ensure_metadata_file()
    
    # Load metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata, reinitializing: {e}")
        # Initialize with default structure
        metadata = {}
        for company_code in COMPANY_FOLDERS:
            metadata[company_code] = {
                "available_years": [],
                "quarters_by_year": {},
                "missing_quarters": {},
                "file_mapping": {},  # New structure to map year/quarter to filenames
                "raw_files": {},
                "fiscal_calendar": {
                    "Q1": "Jun",  # Default fiscal calendar
                    "Q2": "Sep",
                    "Q3": "Dec",
                    "Q4": "Mar"
                },
                "last_updated": datetime.now().isoformat()
            }
        
        # Update metadata with processed data
        for company_code, data in processed_data.items():
            # Initialize company data if not exists
            if company_code not in metadata:
                metadata[company_code] = {
                    "available_years": [],
                    "quarters_by_year": {},
                    "missing_quarters": {},
                    "file_mapping": {},  # New structure to map year/quarter to filenames
                    "raw_files": {},
                    "fiscal_calendar": {
                        "Q1": "Jun",  # Default fiscal calendar
                        "Q2": "Sep",
                        "Q3": "Dec",
                        "Q4": "Mar"
                    },
                    "last_updated": datetime.now().isoformat()
                }
            
            # Update available years
            years_processed = data.get("years_processed", [])
            metadata[company_code]["available_years"] = sorted(list(set(
                metadata[company_code]["available_years"] + years_processed
            )))
            
            # Update quarters by year
            all_quarters = ["Q1", "Q2", "Q3", "Q4"]
            quarters_processed = data.get("quarters_processed", {})
            
            for year, quarters in quarters_processed.items():
                if quarters:  # Only update if quarters were actually found
                    if year not in metadata[company_code]["quarters_by_year"]:
                        metadata[company_code]["quarters_by_year"][year] = []
                    
                    # Add new quarters
                    for quarter in quarters:
                        if quarter not in metadata[company_code]["quarters_by_year"][year]:
                            metadata[company_code]["quarters_by_year"][year].append(quarter)
                    
                    # Sort quarters
                    metadata[company_code]["quarters_by_year"][year] = sorted(
                        metadata[company_code]["quarters_by_year"][year]
                    )
            
            # Update missing quarters for each year
            for year in years_processed:
                year_str = str(year)
                available_quarters = metadata[company_code]["quarters_by_year"].get(year_str, [])
                
                if year_str not in metadata[company_code]["missing_quarters"]:
                    metadata[company_code]["missing_quarters"][year_str] = []
                
                # Clear previous missing quarters for this year
                metadata[company_code]["missing_quarters"][year_str] = []
                
                # Identify missing quarters
                for q in all_quarters:
                    if q not in available_quarters:
                        metadata[company_code]["missing_quarters"][year_str].append(q)
            
            # Update timestamp
            metadata[company_code]["last_updated"] = datetime.now().isoformat()
        
        # Add file mapping for processed reports
        # Scan the dataset folders to map year/quarter to actual files
        for company_code in processed_data.keys():
            dataset_folder = DATASET_FOLDERS.get(company_code)
            if not dataset_folder or not os.path.exists(dataset_folder):
                continue
                
            # Get all JSON files in the dataset folder
            json_files = [f for f in os.listdir(dataset_folder) 
                         if f.lower().endswith('.json') and not f.startswith('.')]
            
            # Initialize file mapping if not exists
            if "file_mapping" not in metadata[company_code]:
                metadata[company_code]["file_mapping"] = {}
                
            # Process each JSON file
            for file_name in json_files:
                file_path = os.path.join(dataset_folder, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract year and quarter from the data
                    year = data.get("year")
                    quarter = data.get("quarter")
                    
                    if year and quarter:
                        year_str = str(year)
                        
                        # Create nested structure if needed
                        if year_str not in metadata[company_code]["file_mapping"]:
                            metadata[company_code]["file_mapping"][year_str] = {}
                            
                        if quarter not in metadata[company_code]["file_mapping"][year_str]:
                            metadata[company_code]["file_mapping"][year_str][quarter] = []
                        
                        # Add file to the mapping if not already there
                        if file_name not in metadata[company_code]["file_mapping"][year_str][quarter]:
                            metadata[company_code]["file_mapping"][year_str][quarter].append(file_name)
                except Exception as e:
                    print(f"Error processing file {file_name} for mapping: {e}")
        
        # Save updated metadata
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("Updated reports metadata file")
            
            # Print information about available and missing quarters
            for company_code in processed_data.keys():
                available = metadata[company_code].get("quarters_by_year", {})
                missing = metadata[company_code].get("missing_quarters", {})
                file_mapping = metadata[company_code].get("file_mapping", {})
                print(f"{company_code} - Available quarters: {available}")
                print(f"{company_code} - Missing quarters: {missing}")
                print(f"{company_code} - File mapping: {file_mapping}")
        except Exception as e:
            print(f"Error saving metadata: {e}")

def process_company_reports():
    """Process all PDF reports for a company and extract financial metrics"""
    
    # Initialize OpenAI LLM
    llm = OpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    
    
    # Create query engine for this company's reports
    query_engine_REXP = create_query_engine("REXP_Reports")
    query_engine_DIPD = create_query_engine("DIPD_Reports")
    
    # Only create tools if query engines were successfully created
    tools = []

    if query_engine_REXP:
        # Create a tool for REXP reports
        query_engine_tool_REXP = QueryEngineTool(
            query_engine=query_engine_REXP,
            metadata=ToolMetadata(
                name="REXP_financial_reports",
                description="Provides information about REXP financial reports for three years"
            )
        )
        tools.append(query_engine_tool_REXP)
    
    if query_engine_DIPD:
        # Create a tool for DIPD reports
        query_engine_tool_DIPP = QueryEngineTool(
            query_engine=query_engine_DIPD,
            metadata=ToolMetadata(
                name="DIPD_financial_reports",
                description="Provides information about DIPD financial reports for three years"
            )
        )
        tools.append(query_engine_tool_DIPP)
    
    if not tools:
        print("No query engines available. Please ensure reports are downloaded.")
        return None

    agent = OpenAIAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        prompt=single_report_prompt
    )

    response = agent.chat(
        """Find the financial metrics for REXP.N0000 and provide as JSON object for each quarter for 2023 and 2024, specifically for the following periods:
        1. Three months ended 30th June 2024
2. Six months ended 30th September 2024
3. Nine months ended 31st December 2023
4. Nine months ended 31st December 2024
        
        Please provide the data in a structured JSON format with all available financial metrics."""
    )
    print(response)
    return response

def parse_quarter_response(response_text):
    """Parse the quarter and year from GPT's response text."""
    match = re.search(r'QUARTER:\s*(Q[1-4])\s*\|\s*YEAR:\s*(\d{4})\s*\|\s*DATE:\s*(\d{4}-\d{2}-\d{2})', response_text, re.IGNORECASE)
    if not match:
        return None, None
    quarter, year, _ = match.groups()
    return quarter.upper(), year

def verify_with_llm(pdf_path, detected_quarter, detected_year):
    """
    Verify a detected quarter and year using LLM.
    
    Args:
        pdf_path: Path to the PDF file
        detected_quarter: Quarter detected by regex patterns (can be None)
        detected_year: Year detected by regex patterns (can be None)
        
    Returns:
        dict: Verification result with keys:
            - is_correct: True if detected values are correct, False otherwise
            - quarter: Suggested quarter (if is_correct is False)
            - year: Suggested year (if is_correct is False)
            - confidence: HIGH/MEDIUM/LOW
            - period_end_date: Date in YYYY-MM-DD format
    """
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return {
            "is_correct": False,
            "quarter": None,
            "year": None,
            "confidence": "LOW",
            "period_end_date": None
        }
    
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(model="gpt-4", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        
        # Create a query engine for this PDF
        query_engine = create_query_engine_for_single_file(pdf_path)
        if not query_engine:
            print(f"Could not create query engine for {os.path.basename(pdf_path)}")
            return {
                "is_correct": False,
                "quarter": None,
                "year": None,
                "confidence": "LOW",
                "period_end_date": None
            }
        
        # Create a query engine tool
        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="pdf_analyzer",
                description=f"Analyzes PDF content from {os.path.basename(pdf_path)}"
            )
        )
        
        # Create an agent with the query engine tool
        agent = OpenAIAgent.from_tools(
            [query_engine_tool],
            llm=llm,
            verbose=False
        )
        
        # Dynamic prompt based on whether we're verifying or detecting from scratch
        if detected_quarter and detected_year:
            prompt_text = f"""You are an expert in financial report analysis.

Analyze this Sri Lankan quarterly financial report and VERIFY if the following detection is correct:
- Detected Quarter: {detected_quarter}
- Detected Year: {detected_year}

Key patterns to look for:
1. "X months ended [DATE]" where X can be 3, 6, 9, or 12
2. "Quarter ended [DATE]"
3. Period end dates in various formats
4. Look for dates in both text and financial statements

Quarter mapping:
- Q1: ends on June 30
- Q2: ends on September 30
- Q3: ends on December 31
- Q4: ends on March 31 (use previous year)

Respond with a JSON containing these fields:
{{
  "is_correct": true/false,
  "quarter": "Q1/Q2/Q3/Q4",
  "year": "YYYY",
  "confidence": "HIGH/MEDIUM/LOW",
  "period_end_date": "YYYY-MM-DD"
}}

If the detection is correct, set is_correct to true and use the detected values.
If incorrect, set is_correct to false and provide the correct values.
If unsure, set confidence to "LOW".
"""
        else:
            prompt_text = """You are an expert in financial report analysis.

Analyze this Sri Lankan quarterly financial report and extract the following:
- Quarter (Q1, Q2, Q3, Q4 only)
- Year (4-digit)
- Period End Date (YYYY-MM-DD)

Key patterns to look for:
1. "X months ended [DATE]" where X can be 3, 6, 9, or 12
2. "Quarter ended [DATE]"
3. Period end dates in various formats
4. Look for dates in both text and financial statements

Quarter mapping:
- Q1: ends on June 30
- Q2: ends on September 30
- Q3: ends on December 31
- Q4: ends on March 31 (use previous year)

Respond with a JSON containing these fields:
{
  "is_correct": true,
  "quarter": "Q1/Q2/Q3/Q4",
  "year": "YYYY",
  "confidence": "HIGH/MEDIUM/LOW",
  "period_end_date": "YYYY-MM-DD"
}

Set is_correct to true since this is a new detection.
If unsure about any value, set confidence to "LOW".
"""
        
        # Get response from LLM
        try:
            response = agent.chat(prompt_text)
            
            # Print raw response for debugging
            print(f"LLM verification response for {os.path.basename(pdf_path)}: {str(response).strip()}")
            
            # Guard clause for empty responses
            response_text = str(response).strip()
            if not response_text or len(response_text) < 5:
                print(f"Empty response from LLM for {os.path.basename(pdf_path)}")
                return {
                    "is_correct": False,
                    "quarter": None,
                    "year": None,
                    "confidence": "LOW",
                    "period_end_date": None
                }
            
            # Try to parse JSON from the response
            try:
                # First try to extract JSON from code blocks if present
                json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to extract just a JSON object from anywhere in the response
                    json_match = re.search(r'({[\s\S]*?})', response_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If all else fails, use the whole response
                        json_str = response_text
                
                # Preprocess the JSON string
                json_str = preprocess_json_string(json_str)
                
                # Parse the JSON
                verification_result = json.loads(json_str)
                
                # Ensure all required fields are in the result
                required_fields = ["is_correct", "quarter", "year", "confidence", "period_end_date"]
                for field in required_fields:
                    if field not in verification_result:
                        if field == "is_correct":
                            verification_result["is_correct"] = True
                        elif field == "confidence":
                            verification_result["confidence"] = "MEDIUM"
                        else:
                            verification_result[field] = None
                
                # Validate the date and quarter mapping if we have a date
                if verification_result["period_end_date"]:
                    try:
                        period_date = datetime.strptime(verification_result["period_end_date"], "%Y-%m-%d")
                        month = period_date.month
                        
                        # Validate quarter matches period end date
                        quarter = verification_result["quarter"]
                        if quarter:
                            expected_quarter = None
                            if month == 6:
                                expected_quarter = "Q1"
                            elif month == 9:
                                expected_quarter = "Q2"
                            elif month == 12:
                                expected_quarter = "Q3"
                            elif month == 3:
                                expected_quarter = "Q4"
                                # For Q4 reports ending in March, adjust year if needed
                                if quarter == "Q4" and verification_result["year"]:
                                    year_int = int(verification_result["year"])
                                    if year_int == period_date.year:
                                        verification_result["year"] = str(period_date.year - 1)
                            
                            # If the quarters don't match what we expect, lower confidence
                            if expected_quarter and quarter != expected_quarter:
                                verification_result["confidence"] = "LOW"
                                print(f"Warning: Quarter {quarter} doesn't match expected quarter {expected_quarter} for end date {verification_result['period_end_date']}")
                    except (ValueError, TypeError) as e:
                        print(f"Error validating date in verification: {e}")
                        verification_result["confidence"] = "LOW"
                
                return verification_result
                
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Error parsing verification response: {e}")
                # Fallback to parsing using regex if JSON parsing fails
                try:
                    # Try old format as fallback
                    match = re.search(r'QUARTER:\s*(Q[1-4])\s*\|\s*YEAR:\s*(\d{4})\s*\|\s*DATE:\s*(\d{4}-\d{2}-\d{2})', response_text, re.IGNORECASE)
                    if match:
                        quarter, year, date_str = match.groups()
                        return {
                            "is_correct": True if (detected_quarter == quarter and detected_year == year) else False,
                            "quarter": quarter.upper(),
                            "year": year,
                            "confidence": "MEDIUM",
                            "period_end_date": date_str
                        }
                except Exception:
                    pass
                
                return {
                    "is_correct": False,
                    "quarter": None,
                    "year": None,
                    "confidence": "LOW",
                    "period_end_date": None
                }
        
        except Exception as e:
            print(f"Error processing LLM verification: {e}")
            return {
                "is_correct": False,
                "quarter": None,
                "year": None,
                "confidence": "LOW",
                "period_end_date": None
            }
            
    except Exception as e:
        print(f"Error in LLM verification: {e}")
        return {
            "is_correct": False,
            "quarter": None,
            "year": None,
            "confidence": "LOW",
            "period_end_date": None
        }

def extract_quarter_from_pdf_content(pdf_path):
    """
    Extract quarter and year information by reading the PDF content.
    First attempts with regex patterns, then ALWAYS verifies with LLM.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        tuple: (quarter, year) or (None, None) if detection fails
    """
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None, None
    
    try:
        # STEP 1: Extract text and detect quarter/year using regex patterns
        regex_quarter = None
        regex_year = None
        
        try:
            # Parse PDF content
            reader = PdfReader(pdf_path)
            text = ""
            
            # Read first 3 pages with different extraction modes
            for page_num in range(min(3, len(reader.pages))):
                page = reader.pages[page_num]
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Text extraction failed for page {page_num}: {e}")
                    continue
            
            text = text.lower()
            
            # Enhanced patterns for quarter detection
            quarter_patterns = {
                "Q1": [
                    r"(?:three|3)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:30(?:th)?\s*)?june",
                    r"first\s*quarter",
                    r"quarter\s*(?:ended|ending)\s*(?:30(?:th)?\s*)?june",
                    r"june\s*quarter",
                    r"q1"
                ],
                "Q2": [
                    r"(?:six|6)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:30(?:th)?\s*)?(?:sep|september)",
                    r"(?:three|3)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:30(?:th)?\s*)?(?:sep|september)",
                    r"second\s*quarter",
                    r"quarter\s*(?:ended|ending)\s*(?:30(?:th)?\s*)?(?:sep|september)",
                    r"september\s*quarter",
                    r"q2"
                ],
                "Q3": [
                    r"(?:nine|9)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:31(?:st)?\s*)?(?:dec|december)",
                    r"(?:three|3)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:31(?:st)?\s*)?(?:dec|december)",
                    r"third\s*quarter",
                    r"quarter\s*(?:ended|ending)\s*(?:31(?:st)?\s*)?(?:dec|december)",
                    r"december\s*quarter",
                    r"q3"
                ],
                "Q4": [
                    r"(?:twelve|12)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:31(?:st)?\s*)?(?:mar|march)",
                    r"(?:three|3)\s*months?\s*(?:period)?\s*end(?:ed|ing)\s*(?:31(?:st)?\s*)?(?:mar|march)",
                    r"fourth\s*quarter",
                    r"quarter\s*(?:ended|ending)\s*(?:31(?:st)?\s*)?(?:mar|march)",
                    r"march\s*quarter",
                    r"q4"
                ]
            }
            
            # Try to find quarter with regex
            for q, patterns in quarter_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                        regex_quarter = q
                        break
                if regex_quarter:
                    break
            
            # Enhanced year detection with regex
            if regex_quarter:
                # Look for year in proximity of quarter mention
                year_pattern = r"(?:20\d{2})"
                for line in text.split('\n'):
                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in quarter_patterns[regex_quarter]):
                        year_match = re.search(year_pattern, line)
                        if year_match:
                            regex_year = year_match.group(0)
                            break
            
            # If year not found near quarter, try other common patterns
            if not regex_year:
                # Try to find year in common date formats
                date_patterns = [
                    r"(?:period|year|quarter).*?(?:end(?:ed|ing)).*?(20\d{2})",
                    r"(?:31(?:st)?|30(?:th)?)\s*(?:march|june|september|december)\s*(20\d{2})",
                    r"(20\d{2}).*?(?:31(?:st)?|30(?:th)?)\s*(?:march|june|september|december)",
                    r"interim.*?(?:report|statement).*?(20\d{2})",
                    r"(20\d{2}).*?interim.*?(?:report|statement)"
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        regex_year = match.group(1)
                        break
        except Exception as e:
            print(f"Error during regex detection: {e}")
        
        # If we found quarter or year with regex, log for diagnostic purposes
        if regex_quarter or regex_year:
            # Special handling for Q4 reports ending in March
            if regex_quarter == "Q4" and regex_year and re.search(r"march|mar", text, re.IGNORECASE):
                # For Q4 reports ending in March, use previous year
                regex_year = str(int(regex_year) - 1)
            
            print(f"Regex detected: Quarter={regex_quarter or 'None'}, Year={regex_year or 'None'} for {os.path.basename(pdf_path)}")
        else:
            print(f"Regex detection failed for {os.path.basename(pdf_path)}")
        
        # STEP 2: ALWAYS verify with LLM
        print(f"Verifying with LLM for {os.path.basename(pdf_path)}...")
        verification_result = verify_with_llm(pdf_path, regex_quarter, regex_year)
        
        # STEP 3: Handle the result based on LLM verification
        if verification_result["is_correct"] and regex_quarter and regex_year:
            # LLM confirms regex detection is correct
            print(f"LLM verified: Quarter={regex_quarter}, Year={regex_year} for {os.path.basename(pdf_path)}")
            return regex_quarter, regex_year
            
        elif verification_result["quarter"] and verification_result["year"]:
            # LLM provides different results
            llm_quarter = verification_result["quarter"]
            llm_year = verification_result["year"]
            confidence = verification_result["confidence"]
            
            if confidence in ["HIGH", "MEDIUM"]:
                # Accept LLM's suggestion with high/medium confidence
                if regex_quarter and regex_year:
                    print(f"LLM override ({confidence} confidence): Changed from {regex_quarter}/{regex_year} to {llm_quarter}/{llm_year} for {os.path.basename(pdf_path)}")
                else:
                    print(f"LLM detected ({confidence} confidence): Quarter={llm_quarter}, Year={llm_year} for {os.path.basename(pdf_path)}")
                return llm_quarter, llm_year
            else:
                # Low confidence - use regex if available, otherwise use LLM result but log warning
                if regex_quarter and regex_year:
                    print(f"Using regex detection (LLM has LOW confidence): Quarter={regex_quarter}, Year={regex_year} for {os.path.basename(pdf_path)}")
                    return regex_quarter, regex_year
                elif llm_quarter and llm_year:
                    print(f"WARNING: Using LLM detection with LOW confidence: Quarter={llm_quarter}, Year={llm_year} for {os.path.basename(pdf_path)}")
                    return llm_quarter, llm_year
        
        # If we get here, neither regex nor LLM could determine both quarter and year
        print(f"Failed to determine quarter and year for {os.path.basename(pdf_path)} using both regex and LLM")
        return None, None
        
    except Exception as e:
        print(f"Error in quarter/year detection for {os.path.basename(pdf_path)}: {e}")
        return None, None

def scan_company_reports(company_code):
    """
    Scan downloaded reports to identify and register them in metadata
    without fully processing them.
    
    Args:
        company_code: The company code to scan reports for
        
    Returns:
        Dictionary with summary of the scan
    """
    company_folder = COMPANY_FOLDERS.get(company_code)
    if not company_folder or not os.path.exists(company_folder):
        return {
            "status": "error",
            "message": f"No reports folder found for {company_code}"
        }
    
    # Get current year for validation
    current_year = datetime.now().year
    valid_years = list(range(current_year - 3, current_year))  # Only last 3 years
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(company_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return {
            "status": "error",
            "message": f"No PDF files found for {company_code}"
        }
    
    # Ensure the metadata file exists with valid structure
    ensure_metadata_file()
    
    # Load or initialize metadata
    metadata_file = "reports_metadata.json"
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata, reinitializing: {e}")
        metadata = {}
        for code in COMPANY_FOLDERS:
            metadata[code] = {
                "available_years": [],
                "quarters_by_year": {},
                "missing_quarters": {},
                "raw_files": {"unknown": []},
                "file_mapping": {},
                "fiscal_calendar": {
                    "Q1": "Jun",
                    "Q2": "Sep",
                    "Q3": "Dec",
                    "Q4": "Mar"
                },
                "last_updated": datetime.now().isoformat()
            }
    
    # Initialize tracking variables
    years_found = []
    total_files = 0
    processed_files = {}
    analysis_results = {
        "successfully_classified": [],
        "still_unknown": [],
        "errors": []
    }
    
    # Initialize raw_files structure if not exists
    if "raw_files" not in metadata[company_code]:
        metadata[company_code]["raw_files"] = {"unknown": []}
    elif "unknown" not in metadata[company_code]["raw_files"]:
        metadata[company_code]["raw_files"]["unknown"] = []
    
    # Create a mapping of already classified files to prevent reprocessing
    already_classified = {}
    for year, quarters in metadata[company_code]["raw_files"].items():
        if year != "unknown":
            for quarter, files in quarters.items():
                for file_name in files:
                    already_classified[file_name] = {"year": year, "quarter": quarter}
    
    # Determine which files need to be processed
    files_to_process = []
    for file_name in pdf_files:
        if file_name in already_classified:
            # File already classified, add to processed_files
            classification = already_classified[file_name]
            year = classification["year"]
            quarter = classification["quarter"]
            
            if year not in processed_files:
                processed_files[year] = {}
            if quarter not in processed_files[year]:
                processed_files[year][quarter] = []
            processed_files[year][quarter].append(file_name)
            
            # Add to successful classification results
            analysis_results["successfully_classified"].append({
                "file": file_name,
                "quarter": quarter,
                "year": year,
                "confidence": "HIGH",
                "evidence": "Previously classified"
            })
            
            print(f"Skipping already classified file: {file_name} (Year={year}, Quarter={quarter})")
        else:
            # New file needs to be processed
            files_to_process.append(file_name)
    
    print(f"Found {len(files_to_process)} new files to process")
    
    # Process each new PDF file
    for file_name in files_to_process:
        total_files += 1
        file_path = os.path.join(company_folder, file_name)
        
        # Extract quarter and year using LLM
        quarter, year = extract_quarter_from_pdf_content(file_path)
        
        if quarter and year and year.isdigit():
            year_int = int(year)
            
            # Validate year is within acceptable range (last 3 years)
            if year_int not in valid_years:
                print(f"Skipping file {file_name} - year {year} not in valid range {min(valid_years)}-{max(valid_years)}")
                # Add to unknown if not in valid range
                if file_name not in metadata[company_code]["raw_files"]["unknown"]:
                    metadata[company_code]["raw_files"]["unknown"].append(file_name)
                continue
            
            # Add to processed files
            if year not in processed_files:
                processed_files[year] = {}
            if quarter not in processed_files[year]:
                processed_files[year][quarter] = []
            processed_files[year][quarter].append(file_name)
            
            # Update metadata structures
            if year not in metadata[company_code]["available_years"]:
                metadata[company_code]["available_years"].append(year)
            
            if year not in years_found:
                years_found.append(year)
            
            if year not in metadata[company_code]["quarters_by_year"]:
                metadata[company_code]["quarters_by_year"][year] = []
            
            if quarter not in metadata[company_code]["quarters_by_year"][year]:
                metadata[company_code]["quarters_by_year"][year].append(quarter)
            
            # Update raw files mapping
            if year not in metadata[company_code]["raw_files"]:
                metadata[company_code]["raw_files"][year] = {}
            
            if quarter not in metadata[company_code]["raw_files"][year]:
                metadata[company_code]["raw_files"][year][quarter] = []
            
            if file_name not in metadata[company_code]["raw_files"][year][quarter]:
                metadata[company_code]["raw_files"][year][quarter].append(file_name)
                
            # Remove from unknown if it was there
            if file_name in metadata[company_code]["raw_files"]["unknown"]:
                metadata[company_code]["raw_files"]["unknown"].remove(file_name)
                
            # Add to successful classification results
            analysis_results["successfully_classified"].append({
                "file": file_name,
                "quarter": quarter,
                "year": year,
                "confidence": "HIGH",  # LLM results are validated
                "evidence": "Detected by LLM"
            })
        else:
            print(f"Could not determine quarter and year for {file_name}")
            # Add to unknown category if not already there
            if file_name not in metadata[company_code]["raw_files"]["unknown"]:
                metadata[company_code]["raw_files"]["unknown"].append(file_name)
            
            # Add to still unknown results
            analysis_results["still_unknown"].append({
                "file": file_name,
                "reason": "LLM detection failed"
            })
    
    # Sort all lists in metadata
    metadata[company_code]["available_years"] = sorted(metadata[company_code]["available_years"])
    
    for year in metadata[company_code]["quarters_by_year"]:
        metadata[company_code]["quarters_by_year"][year] = sorted(metadata[company_code]["quarters_by_year"][year])
    
    # Update missing quarters
    for year in metadata[company_code]["available_years"]:
        available_quarters = metadata[company_code]["quarters_by_year"].get(year, [])
        
        if year not in metadata[company_code]["missing_quarters"]:
            metadata[company_code]["missing_quarters"][year] = []
        
        # Clear previous missing quarters
        metadata[company_code]["missing_quarters"][year] = []
        
        # Identify missing quarters
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q not in available_quarters:
                metadata[company_code]["missing_quarters"][year].append(q)
    
    # Remove years outside the valid range
    metadata[company_code]["available_years"] = [
        year for year in metadata[company_code]["available_years"]
        if int(year) in valid_years
    ]
    
    # Clean up metadata for years outside valid range
    years_to_remove = []
    for year in metadata[company_code]["quarters_by_year"]:
        if int(year) not in valid_years:
            years_to_remove.append(year)
    
    for year in years_to_remove:
        del metadata[company_code]["quarters_by_year"][year]
        if year in metadata[company_code]["missing_quarters"]:
            del metadata[company_code]["missing_quarters"][year]
        if year in metadata[company_code]["raw_files"]:
            del metadata[company_code]["raw_files"][year]
    
    # Update timestamp
    metadata[company_code]["last_updated"] = datetime.now().isoformat()
    
    # Save updated metadata
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata for {company_code}")
    except Exception as e:
        print(f"Error saving metadata: {e}")
    
    # Return summary
    return {
        "status": "success",
        "years_found": sorted(years_found),
        "total_files": total_files,
        "available_years": sorted(metadata[company_code]["available_years"]),
        "quarters_by_year": metadata[company_code]["quarters_by_year"],
        "missing_quarters": metadata[company_code]["missing_quarters"],
        "processed_files": processed_files,
        "analysis_results": analysis_results
    }

def analyze_unknown_reports(company_code):
    """
    Analyze unknown reports using LLM to determine their quarter and year.
    Updates the metadata file with the findings.
    
    Args:
        company_code: The company code to analyze unknown reports for
        
    Returns:
        Dictionary with analysis results
    """
    # Get company folder
    company_folder = COMPANY_FOLDERS.get(company_code)
    if not company_folder or not os.path.exists(company_folder):
        return {
            "status": "error",
            "message": f"No reports folder found for {company_code}"
        }
    
    # Load current metadata
    metadata_file = "reports_metadata.json"
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {"status": "error", "message": "Could not load metadata"}
    
    if company_code not in metadata:
        return {"status": "error", "message": "Company not found in metadata"}
    
    # Get unknown files
    unknown_files = metadata.get(company_code, {}).get("raw_files", {}).get("unknown", [])
    if not unknown_files:
        return {"status": "success", "message": "No unknown files to analyze"}
    
    print(f"Starting analysis of {len(unknown_files)} unknown files for {company_code}")
    
    # Get current year for validation
    current_year = datetime.now().year
    valid_years = list(range(current_year - 3, current_year))  # Only last 3 years
    
    analysis_results = {
        "successfully_classified": [],
        "still_unknown": [],
        "errors": []
    }
    
    # Create a copy of unknown_files to safely modify during iteration
    unknown_files_to_process = unknown_files.copy()
    
    for file_name in unknown_files_to_process:
        file_path = os.path.join(company_folder, file_name)
        if not os.path.exists(file_path):
            analysis_results["errors"].append({
                "file": file_name,
                "error": "File not found"
            })
            continue
        
        print(f"Analyzing unknown file: {file_name}")
        
        try:
            # Use the LLM to get quarter and year
            quarter, year = extract_quarter_from_pdf_content(file_path)
            
            if quarter and year and year.isdigit():
                year_int = int(year)
                
                # Validate year is within acceptable range (last 3 years)
                if year_int not in valid_years:
                    print(f"Year {year} for file {file_name} not in valid range {min(valid_years)}-{max(valid_years)}")
                    analysis_results["still_unknown"].append({
                        "file": file_name,
                        "reason": f"Year {year} not in valid range"
                    })
                    continue
                
                # Update metadata structures
                # Remove from unknown
                if file_name in metadata[company_code]["raw_files"]["unknown"]:
                    metadata[company_code]["raw_files"]["unknown"].remove(file_name)
                
                # Add to correct year and quarter
                if year not in metadata[company_code]["raw_files"]:
                    metadata[company_code]["raw_files"][year] = {}
                if quarter not in metadata[company_code]["raw_files"][year]:
                    metadata[company_code]["raw_files"][year][quarter] = []
                if file_name not in metadata[company_code]["raw_files"][year][quarter]:
                    metadata[company_code]["raw_files"][year][quarter].append(file_name)
                
                # Update available years
                if year not in metadata[company_code]["available_years"]:
                    metadata[company_code]["available_years"].append(year)
                    metadata[company_code]["available_years"].sort()
                
                # Update quarters by year
                if year not in metadata[company_code]["quarters_by_year"]:
                    metadata[company_code]["quarters_by_year"][year] = []
                if quarter not in metadata[company_code]["quarters_by_year"][year]:
                    metadata[company_code]["quarters_by_year"][year].append(quarter)
                    metadata[company_code]["quarters_by_year"][year].sort()
                
                # Update missing quarters
                if year not in metadata[company_code]["missing_quarters"]:
                    metadata[company_code]["missing_quarters"][year] = []
                metadata[company_code]["missing_quarters"][year] = [
                    q for q in ["Q1", "Q2", "Q3", "Q4"]
                    if q not in metadata[company_code]["quarters_by_year"][year]
                ]
                
                analysis_results["successfully_classified"].append({
                    "file": file_name,
                    "quarter": quarter,
                    "year": year,
                    "confidence": "HIGH",  # All LLM responses are validated
                    "evidence": "Detected by LLM"
                })
                
                print(f"Successfully classified {file_name} as {quarter} {year}")
            else:
                analysis_results["still_unknown"].append({
                    "file": file_name,
                    "reason": "LLM detection failed"
                })
                print(f"Failed to classify {file_name} - LLM detection returned no valid quarter/year")
        except Exception as e:
            analysis_results["errors"].append({
                "file": file_name,
                "error": str(e)
            })
            print(f"Error analyzing {file_name}: {str(e)}")
    
    # Check for duplicate files across different classifications
    duplicate_check = {}
    for year, quarters in metadata[company_code]["raw_files"].items():
        if year == "unknown":
            continue
        for quarter, files in quarters.items():
            for file_name in files:
                if file_name not in duplicate_check:
                    duplicate_check[file_name] = []
                duplicate_check[file_name].append(f"{year}/{quarter}")
    
    # Report and clean up duplicates
    for file_name, locations in duplicate_check.items():
        if len(locations) > 1:
            print(f"Warning: {file_name} is classified in multiple locations: {', '.join(locations)}")
            # Keep the most recent classification only (we won't remove here to avoid complexity)
    
    # Update metadata file
    try:
        metadata[company_code]["last_updated"] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated metadata for {company_code} after analyzing {len(unknown_files_to_process)} unknown files")
    except Exception as e:
        print(f"Error saving metadata: {e}")
        analysis_results["errors"].append({
            "error": f"Could not save metadata: {str(e)}"
        })
    
    # Provide a summary
    summary = {
        "total_unknown": len(unknown_files_to_process),
        "successfully_classified": len(analysis_results["successfully_classified"]),
        "still_unknown": len(analysis_results["still_unknown"]),
        "errors": len(analysis_results["errors"]),
        "remaining_unknown": len(metadata[company_code]["raw_files"]["unknown"])
    }
    print(f"Analysis summary: {summary}")
    
    return {
        "status": "success",
        "results": analysis_results,
        "summary": summary
    }

def preprocess_json_string(json_str: str) -> str:
    """
    Preprocess a JSON string to handle common formatting issues:
    1. Remove commas from numeric values
    2. Handle truncated JSON strings
    3. Fix common formatting issues
    
    Args:
        json_str: The JSON string to preprocess
        
    Returns:
        Preprocessed JSON string or empty string if preprocessing fails
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return "{}"
        
    try:
        # First try to parse as is - if it works, return original
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # If parsing fails, apply preprocessing
        try:
            # Remove commas between digits in numeric values
            def replace_commas_in_numbers(match):
                # Only replace commas between digits
                number = match.group(0)
                if '.' in number:
                    # Handle decimal numbers
                    parts = number.split('.')
                    if len(parts) == 2:
                        integer_part = parts[0].replace(',', '')
                        return f"{integer_part}.{parts[1]}"
                return number.replace(',', '')
            
            # Pattern for numbers with commas (both positive and negative)
            number_pattern = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
            processed_str = re.sub(number_pattern, replace_commas_in_numbers, json_str)
            
            # Check if JSON string is truncated
            if processed_str.count('{') > processed_str.count('}'):
                # Try to find the last complete object
                last_complete = 0
                brace_count = 0
                for i, char in enumerate(processed_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_complete = i + 1
                
                if last_complete > 0:
                    processed_str = processed_str[:last_complete]
                else:
                    # If no complete object found, try to complete it
                    processed_str = processed_str + "}" * (processed_str.count('{') - processed_str.count('}'))
            
            # Remove any trailing commas before closing braces
            processed_str = re.sub(r',(\s*})', r'\1', processed_str)
            
            # Try to parse the preprocessed string
            try:
                # Validate the JSON is now valid
                json.loads(processed_str)
                return processed_str
            except json.JSONDecodeError:
                print("Failed to validate preprocessed JSON")
                return "{}"
            
        except Exception as e:
            print(f"Error during JSON preprocessing: {str(e)}")
            print(f"Problem JSON: {json_str}")
            return "{}"

# Only run process_company_reports when the file is executed directly
if __name__ == "__main__":
    process_company_reports()
