import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.base.response.schema import Response
# from llama_index.core.query_engine import QueryEngine
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define schema for quarterly financial data
class QuarterlyFinancialData(BaseModel):
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(description="Quarter (Q1, Q2, Q3, or Q4)")
    year: str = Field(description="4-digit year")
    fiscal_year: Optional[str] = Field(description="Fiscal year if mentioned (e.g., '2023/24')", default=None)
    revenue: Optional[float] = Field(description="Revenue/Turnover in thousands", default=None)
    cogs: Optional[float] = Field(description="Cost of Goods Sold/Cost of Sales in thousands", default=None)
    gross_profit: Optional[float] = Field(description="Gross Profit in thousands", default=None)
    operating_expenses: Optional[float] = Field(description="Total Operating Expenses in thousands", default=None)
    operating_income: Optional[float] = Field(description="Operating Income/Operating Profit in thousands", default=None)
    finance_income: Optional[float] = Field(description="Finance Income in thousands", default=None)
    finance_cost: Optional[float] = Field(description="Finance Cost in thousands", default=None)
    profit_before_tax: Optional[float] = Field(description="Profit Before Tax in thousands", default=None)
    tax_expense: Optional[float] = Field(description="Tax Expense in thousands", default=None)
    net_income: Optional[float] = Field(description="Net Income/Profit after tax in thousands", default=None)
    eps: Optional[float] = Field(description="Earnings Per Share", default=None)

class CompanyQuarterlyData(BaseModel):
    company: str = Field(description="Company ticker symbol (e.g., 'REXP.N0000' or 'DIPD.N0000')")
    quarterly_data: List[QuarterlyFinancialData] = Field(description="List of quarterly financial data")

class MultiYearQuarterlyFinancialData(BaseModel):
    rexp_data: CompanyQuarterlyData = Field(description="Richard Pieris PLC (REXP.N0000) quarterly financial data")
    dipd_data: CompanyQuarterlyData = Field(description="Dipped Products PLC (DIPD.N0000) quarterly financial data")

def load_financial_data(folder_path):
    """Load financial data from JSON files"""
    financial_data = []
    
    # Check if folder exists and has files
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return []
    
    # Check if directory has any JSON files
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    if not files:
        print(f"Warning: No JSON files found in {folder_path}.")
        return []
    
    # Load all JSON files
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add file source information
                data['file_source'] = file_name
                financial_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return financial_data

def create_query_engine(folder_path):
    """Create a query engine for a specific company's documents"""
    # Load the financial data from JSON files
    financial_data = load_financial_data(folder_path)
    
    if not financial_data:
        return create_fallback_engine(folder_path)
    
    try:
        # Create documents from financial data
        documents = []
        for data in financial_data:
            # Create a formatted text representation of the financial data
            content = f"""
            Quarter: {data.get('quarter', 'Unknown')}
            Revenue: {data.get('revenue', 'N/A')}
            COGS: {data.get('cogs', 'N/A')}
            Gross Profit: {data.get('gross_profit', 'N/A')}
            Operating Expenses: {data.get('operating_expenses', 'N/A')}
            Operating Income: {data.get('operating_income', 'N/A')}
            Net Income: {data.get('net_income', 'N/A')}
            """
            
            # Create a Document with the content and original data as metadata
            doc = Document(
                text=content,
                metadata={
                    'quarter': data.get('quarter', 'Unknown'),
                    'financial_data': data,
                    'source': data.get('source_document', data.get('file_source', 'Unknown'))
                }
            )
            documents.append(doc)
        
        # Create index from documents
        index = VectorStoreIndex.from_documents(documents)
        
        # Create query engine with improved response synthesis
        query_engine = index.as_query_engine(
            response_mode="compact",
            verbose=True
        )
        
        return query_engine
    except Exception as e:
        print(f"Error creating query engine for {folder_path}: {e}")
        return create_fallback_engine(folder_path)

def create_fallback_engine(folder_path):
    """Create a fallback query engine that returns a message to process reports"""
    class SimpleFallbackQueryEngine:
        def query(self, query_str):
            response_text = f"No financial data available. Please download and process reports first using the buttons in the sidebar.\n\nThe dataset folder '{folder_path}' is empty or doesn't exist."
            return Response(response=response_text)
    
    return SimpleFallbackQueryEngine()

def get_quarterly_data_for_years(company: str, years: int = 3) -> CompanyQuarterlyData:
    """
    Retrieves financial data for all quarters over the specified number of years
    for the given company.
    
    Args:
        company: Company ticker symbol ('REXP.N0000' or 'DIPD.N0000')
        years: Number of years of data to retrieve (default: 3)
        
    Returns:
        CompanyQuarterlyData object containing all quarterly financial data
    """
    folder_path = "REXP_Datasets" if company.startswith("REXP") else "DIPD_Datasets"
    financial_data = load_financial_data(folder_path)
    
    if not financial_data:
        # Return empty data structure if no data available
        return CompanyQuarterlyData(
            company=company,
            quarterly_data=[]
        )
    
    # Sort data by year and quarter
    financial_data.sort(key=lambda x: (x.get('year', '0000'), x.get('quarter', 'Q0')))
    
    # Get the most recent years (up to the number specified)
    unique_years = sorted(set(item.get('year') for item in financial_data if 'year' in item), reverse=True)
    target_years = unique_years[:years] if len(unique_years) > years else unique_years
    
    # Filter data for target years
    filtered_data = [
        item for item in financial_data 
        if item.get('year') in target_years
    ]
    
    # Convert to QuarterlyFinancialData objects
    quarterly_data = []
    for item in filtered_data:
        try:
            quarterly_data.append(QuarterlyFinancialData(
                quarter=item.get('quarter', 'Q1'),
                year=item.get('year', ''),
                fiscal_year=item.get('fiscal_year'),
                revenue=item.get('revenue'),
                cogs=item.get('cogs'),
                gross_profit=item.get('gross_profit'),
                operating_expenses=item.get('operating_expenses'),
                operating_income=item.get('operating_income'),
                finance_income=item.get('finance_income'),
                finance_cost=item.get('finance_cost'),
                profit_before_tax=item.get('profit_before_tax'),
                tax_expense=item.get('tax_expense'),
                net_income=item.get('net_income'),
                eps=item.get('eps')
            ))
        except Exception as e:
            print(f"Error converting financial data: {e}")
    
    return CompanyQuarterlyData(
        company=company,
        quarterly_data=quarterly_data
    )

def get_multi_year_quarterly_data(years: int = 3) -> MultiYearQuarterlyFinancialData:
    """
    Retrieves quarterly financial data for both companies over the specified 
    number of years.
    
    Args:
        years: Number of years of data to retrieve (default: 3)
        
    Returns:
        MultiYearQuarterlyFinancialData object containing quarterly data for both companies
    """
    rexp_data = get_quarterly_data_for_years("REXP.N0000", years)
    dipd_data = get_quarterly_data_for_years("DIPD.N0000", years)
    
    return MultiYearQuarterlyFinancialData(
        rexp_data=rexp_data,
        dipd_data=dipd_data
    )

# Function to be exposed as a tool
def retrieve_quarterly_financial_data(years: int = 3) -> Dict[str, Any]:
    """
    Retrieves and returns quarterly financial data for both Richard Pieris PLC (REXP.N0000)
    and Dipped Products PLC (DIPD.N0000) for the specified number of years.
    
    Args:
        years: Number of years of data to retrieve (default: 3)
        
    Returns:
        Dictionary containing quarterly financial data for both companies
    """
    multi_year_data = get_multi_year_quarterly_data(years)
    return multi_year_data.dict()

# Define Pydantic schema for function input parameters
class QueryFinancialDataSchema(BaseModel):
    years: int = Field(
        default=3,
        description="Number of years of data to retrieve (1-3)",
        ge=1,
        le=5
    )

# Create a tool for the quarterly financial data retrieval
quarterly_financial_data_tool = FunctionTool.from_defaults(
    fn=retrieve_quarterly_financial_data,
    name="retrieve_quarterly_financial_data",
    description="Retrieves quarterly financial data for Richard Pieris PLC (REXP.N0000) and Dipped Products PLC (DIPD.N0000) 3 years.",
    fn_schema=QueryFinancialDataSchema,
    return_direct=False
)

# Try to create query engines for both companies
try:
    rexp_engine = create_query_engine("REXP_Datasets")
except Exception as e:
    print(f"Error creating REXP engine: {e}")
    rexp_engine = create_fallback_engine("REXP_Datasets")

try:
    dipd_engine = create_query_engine("DIPD_Datasets")
except Exception as e:
    print(f"Error creating DIPD engine: {e}")
    dipd_engine = create_fallback_engine("DIPD_Datasets")

# Create tools for each query engine with improved descriptions
rexp_tool = QueryEngineTool(
    query_engine=rexp_engine,
    metadata=ToolMetadata(
        name="rexp_query",
        description="Use this tool to query Richard Pieris PLC (REXP) financial reports. It can extract quarterly financial metrics like Revenue, COGS, Gross Profit, Operating Expenses, Operating Income, and Net Income."
    )
)

dipd_tool = QueryEngineTool(
    query_engine=dipd_engine,
    metadata=ToolMetadata(
        name="dipd_query",
        description="Use this tool to query Dipped Products PLC (DIPD) financial reports. It can extract quarterly financial metrics like Revenue, COGS, Gross Profit, Operating Expenses, Operating Income, and Net Income."
    )
)

# Create sub-agent with a specialized system prompt for financial analysis
sub_agent = OpenAIAgent.from_tools(
    tools=[rexp_tool, dipd_tool],
    verbose=True,
    system_prompt="""
    You are a specialized financial analysis assistant for companies listed on the Colombo Stock Exchange.
    
    When extracting financial metrics:
    1. Always specify the exact quarter and year (e.g., Q1 2022)
    2. Present numerical data without currency symbols unless specifically asked
    3. Look for consolidated financial statements when available
    4. For comparative data, always ensure time periods match
    5. Identify the reporting currency and units (e.g., thousands)
    6. When asked for specific metrics like Revenue, COGS, Gross Profit, Operating Expenses, Operating Income, or Net Income, extract the precise values
    7. For extracting quarterly data sequences, return values in chronological order
    8. You can use the retrieve_quarterly_financial_data tool to get comprehensive quarterly data for both companies
    
    Focus on accuracy and clarity in your responses.
    """
)

def call_sub_agent(query):
    """Call the sub agent to get the response"""
    response = sub_agent.chat(query)
    return response

call_sub_agent_tool = FunctionTool.from_defaults(fn=call_sub_agent)

# Create the main agent with all tools and enhanced system prompt
agent = OpenAIAgent.from_tools(
    tools=[rexp_tool, dipd_tool, quarterly_financial_data_tool, call_sub_agent_tool],
    verbose=True,
    system_prompt="""
    You are a financial analysis assistant specializing in Colombo Stock Exchange companies.
    
    When analyzing financial data:
    1. Prioritize the most recent quarterly reports
    2. Compare year-over-year performance for each metric
    3. Calculate growth rates and trends when relevant
    4. Focus on key metrics: Revenue, COGS, Gross Profit, Operating Expenses, Operating Income, Net Income
    5. Use the specialized tools available to extract detailed financial information
    6. When asked to extract specific numerical data series, format your response to be easily parsable
    7. For numerical sequences, return them in a clear format: [value1, value2, value3, ...]
    8. You can use the retrieve_quarterly_financial_data tool to get comprehensive quarterly data for both companies
    
    When dealing with financial metrics:
    - Revenue: Look for terms like "Revenue", "Turnover", or "Sales" in income statements
    - COGS: Look for terms like "Cost of Sales" or "Cost of Goods Sold"
    - Gross Profit: The difference between Revenue and COGS
    - Operating Expenses: Look for "Operating Expenses", "Admin Expenses", "Distribution Costs", etc.
    - Operating Income: The difference between Gross Profit and Operating Expenses
    - Net Income: Look for "Net Profit", "Profit After Tax", "Net Income", or "Profit Attributable to Ordinary Shareholders"
    
    Focus on providing accurate, concise, and actionable financial insights.
    """
)










