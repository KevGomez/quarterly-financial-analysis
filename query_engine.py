import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.base.response.schema import Response

# Load environment variables
load_dotenv()

def create_query_engine(folder_path):
    """Create a query engine for a specific company's documents"""
    # Check if folder exists and has files
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return create_fallback_engine(folder_path)
    
    # Check if directory has any text files
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not files:
        print(f"Warning: No files found in {folder_path}.")
        return create_fallback_engine(folder_path)
    
    try:
        # Load documents from the dataset folder
        documents = SimpleDirectoryReader(
            input_dir=folder_path,
            filename_as_id=True
        ).load_data()
        
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

# Try to create query engines for both companies
try:
    rexp_engine = create_query_engine("Company1_Datasets")
except Exception as e:
    print(f"Error creating REXP engine: {e}")
    rexp_engine = create_fallback_engine("Company1_Datasets")

try:
    dipd_engine = create_query_engine("Company2_Datasets")
except Exception as e:
    print(f"Error creating DIPD engine: {e}")
    dipd_engine = create_fallback_engine("Company2_Datasets")

# Create tools for each query engine with improved descriptions
rexp_tool = QueryEngineTool(
    query_engine=rexp_engine,
    metadata=ToolMetadata(
        name="rexp_query",
        description="Use this tool to query Richard Pieris PLC (REXP) financial reports. It can extract quarterly financial metrics like Revenue, Profit, EPS, and Operating Expenses."
    )
)

dipd_tool = QueryEngineTool(
    query_engine=dipd_engine,
    metadata=ToolMetadata(
        name="dipd_query",
        description="Use this tool to query Dipped Products PLC (DIPD) financial reports. It can extract quarterly financial metrics like Revenue, Profit, EPS, and Operating Expenses."
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
    5. Identify the reporting currency and units (e.g., millions, thousands)
    6. When asked for specific metrics like Revenue, Profit, EPS, or Operating Expenses, extract the precise values
    7. For extracting quarterly data sequences, return values in chronological order
    
    Focus on accuracy and clarity in your responses.
    """
)

def call_sub_agent(query):
    """Call the sub agent to get the response"""
    response = sub_agent.chat(query)
    return response

call_sub_agent_tool = FunctionTool.from_defaults(fn=call_sub_agent)

# Create the main agent with both tools and enhanced system prompt
agent = OpenAIAgent.from_tools(
    tools=[rexp_tool, dipd_tool, call_sub_agent_tool],
    verbose=True,
    system_prompt="""
    You are a financial analysis assistant specializing in Colombo Stock Exchange companies.
    
    When analyzing financial data:
    1. Prioritize the most recent quarterly reports
    2. Compare year-over-year performance for each metric
    3. Calculate growth rates and trends when relevant
    4. Focus on key metrics: Revenue, Gross Profit, Operating Expenses, Net Income, EPS
    5. Use the specialized tools available to extract detailed financial information
    6. When asked to extract specific numerical data series, format your response to be easily parsable
    7. For numerical sequences, return them in a clear format: [value1, value2, value3, ...]
    
    When dealing with financial metrics:
    - Revenue: Look for terms like "Revenue", "Turnover", or "Sales" in income statements
    - Profit: Extract "Net Profit", "Profit After Tax", or "Net Income"
    - EPS: Find "Earnings Per Share" or "Basic EPS" values
    - Operating Expenses: Look for "Operating Expenses", "Admin Expenses", "Distribution Costs", etc.
    
    Focus on providing accurate, concise, and actionable financial insights.
    """
)










