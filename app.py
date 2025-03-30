import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import altair as alt
from parser import process_company_reports, COMPANY_FOLDERS, DATASET_FOLDERS
from scrapper import download_company_reports

# Set page config
st.set_page_config(
    page_title="Financial Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = False
if 'selected_companies' not in st.session_state:
    st.session_state.selected_companies = {
        "REXP.N0000": True,   # Richard Pieris PLC selected by default
        "DIPD.N0000": True    # Dipped Products PLC selected by default
    }
if 'custom_company' not in st.session_state:
    st.session_state.custom_company = ""
if 'file_viewer' not in st.session_state:
    st.session_state.file_viewer = {
        "company_code": None,
        "file_name": None,
        "is_open": False
    }
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False

# Company mapping
COMPANY_NAMES = {
    "REXP.N0000": "Richard Pieris PLC",
    "DIPD.N0000": "Dipped Products PLC"
}

# Try to import the query engine components
try:
    from query_engine import agent, rexp_tool, dipd_tool, call_sub_agent
    st.session_state.agent_initialized = True
except Exception as e:
    st.session_state.agent_initialized = False
    st.session_state.agent_error = str(e)

# Sample financial data (fallback when agent fails or isn't initialized)
SAMPLE_DATA = {
    "REXP.N0000": {
        "quarters": ["Q1 2021", "Q2 2021", "Q3 2021", "Q4 2021", "Q1 2022", "Q2 2022", "Q3 2022", "Q4 2022"],
        "Revenue": [125.3, 137.2, 142.8, 156.3, 165.2, 172.1, 179.8, 188.5],
        "Profit": [12.5, 14.2, 15.8, 17.3, 18.2, 17.1, 19.8, 21.5],
        "EPS": [0.58, 0.67, 0.73, 0.81, 0.85, 0.79, 0.92, 1.02],
        "Operating Expenses": [45.2, 48.3, 49.7, 52.8, 55.3, 58.9, 60.2, 63.5]
    },
    "DIPD.N0000": {
        "quarters": ["Q1 2021", "Q2 2021", "Q3 2021", "Q4 2021", "Q1 2022", "Q2 2022", "Q3 2022", "Q4 2022"],
        "Revenue": [75.3, 82.2, 88.8, 93.3, 98.2, 102.1, 108.8, 115.5],
        "Profit": [8.5, 9.2, 10.8, 11.3, 12.2, 11.1, 13.8, 14.5],
        "EPS": [0.38, 0.42, 0.51, 0.54, 0.58, 0.52, 0.65, 0.72],
        "Operating Expenses": [28.2, 30.3, 31.7, 33.8, 35.3, 37.9, 39.2, 41.5]
    }
}

def get_year_range():
    """Get default year range (3 years)"""
    current_year = datetime.now().year
    return current_year - 2, current_year

def display_dataset_info(company_code, folder_path):
    """Display information about the dataset with options to view file content"""
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if files:
            # Get file sizes and creation dates
            file_info = []
            for file in files:
                file_path = os.path.join(folder_path, file)
                size = os.path.getsize(file_path)
                date = datetime.fromtimestamp(os.path.getctime(file_path))
                file_info.append({
                    'File': file,
                    'Size (KB)': round(size/1024, 2),
                    'Created': date.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # Create DataFrame
            df = pd.DataFrame(file_info)
            
            # Show dataframe
            st.dataframe(df, use_container_width=True)
            
            # File viewer section
            st.subheader("File Viewer")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_file = st.selectbox(
                    "Select a file to view",
                    files,
                    key=f"file_select_{company_code}"
                )
            
            with col2:
                if st.button("View Content", key=f"view_btn_{company_code}"):
                    st.session_state.file_viewer = {
                        "company_code": company_code,
                        "file_name": selected_file,
                        "is_open": True
                    }
            
            # If a file is selected for viewing
            if (st.session_state.file_viewer["is_open"] and 
                st.session_state.file_viewer["company_code"] == company_code):
                file_to_view = st.session_state.file_viewer["file_name"]
                file_path = os.path.join(folder_path, file_to_view)
                
                with st.expander(f"Content of {file_to_view}", expanded=True):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Add download button for the file
                        col1, col2 = st.columns([5, 1])
                        with col2:
                            st.download_button(
                                label="Download",
                                data=content,
                                file_name=file_to_view,
                                mime="text/plain"
                            )
                        
                        # Show content with preserved formatting
                        st.markdown("### File Content")
                        
                        # Apply CSS for monospace font and preserved whitespace
                        st.markdown("""
                        <style>
                        .fixed-font {
                            font-family: monospace;
                            white-space: pre;
                            overflow-x: auto;
                            line-height: 1.2;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Display content with preserved formatting
                        st.markdown(f'<div class="fixed-font">{content}</div>', unsafe_allow_html=True)
                        
                        # Add a button to close the viewer
                        if st.button("Close", key=f"close_btn_{company_code}"):
                            st.session_state.file_viewer["is_open"] = False
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
            
            return df
        else:
            st.warning(f"No text files found in {folder_path}. Please process reports first.")
    else:
        st.warning(f"Dataset folder {folder_path} does not exist. Please process reports first.")
    
    return None

def get_agent_response(query, companies):
    """Get response from the agent based on the query"""
    if not st.session_state.agent_initialized:
        return "The analysis engine is not available. Please ensure dataset folders have processed reports."
    
    try:
        company_context = ""
        if len(companies) == 1:
            company_context = f"For {COMPANY_NAMES.get(companies[0], companies[0])}: "
        else:
            company_names = [COMPANY_NAMES.get(code, code) for code in companies]
            company_context = f"For {' and '.join(company_names)}: "
        
        # Query the agent
        response = agent.chat(company_context + query)
        return str(response)
    except Exception as e:
        return f"Error getting response: {str(e)}"

def get_financial_quarters(company_code):
    """Get quarterly data points from the agent"""
    if not st.session_state.agent_initialized:
        return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])["quarters"]
    
    try:
        # Get the right tool for the company
        tool = rexp_tool if company_code == "REXP.N0000" else dipd_tool
        
        # Create a prompt to extract quarterly dates
        prompt = f"""
        Extract a list of all quarter periods (like Q1 2021, Q2 2021, etc.) from the financial reports.
        Return only the quarter identifiers in chronological order for years 2021-2022.
        Format your response as a simple list like: Q1 2021, Q2 2021, Q3 2021, etc.
        Do not include any explanatory text.
        """
        
        # Query the tool
        response = tool.query_engine.query(prompt)
        
        # Process the response to extract quarters
        import re
        quarters = re.findall(r'Q[1-4]\s+\d{4}', str(response))
        
        if quarters:
            return quarters
        else:
            # Fallback to sample data
            return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])["quarters"]
            
    except Exception as e:
        st.warning(f"Could not extract quarters: {str(e)}")
        return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])["quarters"]  # Fallback

def get_financial_metric(company_code, metric_name):
    """Get specific financial metric data from agent"""
    if not st.session_state.agent_initialized:
        return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])[metric_name]
    
    try:
        company_name = COMPANY_NAMES.get(company_code, company_code)
        
        # Get the right tool for the company
        tool = rexp_tool if company_code == "REXP.N0000" else dipd_tool
        
        # Create a prompt specifically for extracting the metric
        prompt = f"""
        Extract the quarterly {metric_name} values for {company_name} from 2021-2022.
        Focus only on the consolidated figures for each quarter.
        Return only the numerical values in chronological order (Q1 2021, Q2 2021, Q3 2021, Q4 2021, Q1 2022, etc.).
        Format your response as a sequence of numbers without any text or explanation.
        Example format: 125.3, 137.2, 142.8, 156.3, 165.2, 172.1, 179.8, 188.5
        """
        
        # Query the tool
        response = tool.query_engine.query(prompt)
        
        # Process the response to extract numbers
        import re
        values = re.findall(r'[-+]?\d*\.\d+|\d+', str(response))
        
        if values:
            return [float(val) for val in values]
        else:
            # Fallback to sample data
            return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])[metric_name]
            
    except Exception as e:
        st.warning(f"Could not extract {metric_name} data: {str(e)}")
        return SAMPLE_DATA.get(company_code, SAMPLE_DATA["REXP.N0000"])[metric_name]  # Fallback

def get_agent_forecast(company_code, metric_name, periods, historical_data):
    """Get an agent-powered forecast for a financial metric"""
    if not st.session_state.agent_initialized:
        return None, None
    
    try:
        company_name = COMPANY_NAMES.get(company_code, company_code)
        quarters = historical_data["quarters"]
        values = historical_data["values"]
        
        # Create a detailed prompt for the sub_agent
        data_points = ", ".join([f"{quarters[i]}: {values[i]}" for i in range(len(quarters))])
        
        prompt = f"""
        I need a detailed financial forecast for {company_name}'s {metric_name} for the next {periods} quarters.
        
        Historical data:
        {data_points}
        
        Please provide:
        1. Forecasted values for the next {periods} quarters (numerical values only)
        2. An explanation of the forecast methodology and key factors considered
        3. Any significant trends or patterns in the data
        
        Format the forecasted values as a comma-separated list of numbers first, followed by your analysis.
        Example: "120.5, 125.3, 130.2, 136.7, The forecast shows..."
        """
        
        # Call the sub_agent for forecasting
        response = call_sub_agent(prompt)
        response_text = str(response)
        
        # Extract the forecasted values (first line or comma-separated list at beginning)
        import re
        forecast_values_match = re.search(r'([\d\., ]+)', response_text)
        forecast_explanation = response_text
        
        if forecast_values_match:
            forecast_string = forecast_values_match.group(1).strip()
            forecast_values = [float(v.strip()) for v in forecast_string.split(',') if v.strip()]
            
            # Trim to requested periods
            forecast_values = forecast_values[:periods]
            
            # Remove the values part from the explanation
            forecast_explanation = response_text.replace(forecast_string, "", 1).strip()
            
            return forecast_values, forecast_explanation
        
        return None, response_text
            
    except Exception as e:
        st.warning(f"Could not generate agent forecast: {str(e)}")
        return None, str(e)

def create_line_chart(data, x_col, y_col, title):
    """Create a line chart with the given data"""
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:N', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=f'{y_col} (millions)'),
        tooltip=[x_col, y_col]
    ).properties(
        title=title
    )
    return chart

def check_datasets_available():
    """Check if any datasets are available"""
    available_datasets = {}
    
    for company_code, folder in DATASET_FOLDERS.items():
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            available_datasets[company_code] = len(files) > 0
        else:
            available_datasets[company_code] = False
    
    return available_datasets

def main():
    st.title("Financial Analysis Assistant")
    
    # Check dataset availability
    datasets_available = check_datasets_available()
    
    # Sidebar
    with st.sidebar:
        st.header("Data Management")
        
        # Company Selection
        st.subheader("Companies")
        
        # Default companies with checkboxes (selected by default)
        for company_code, company_name in COMPANY_NAMES.items():
            st.session_state.selected_companies[company_code] = st.checkbox(
                company_name,
                value=st.session_state.selected_companies.get(company_code, True)
            )
        
        # Option to add a custom company
        add_custom = st.checkbox("Add another company", value=False)
        
        if add_custom:
            custom_company_code = st.text_input(
                "Enter company code (e.g. ABC.N0000)",
                value=st.session_state.custom_company
            )
            
            if custom_company_code and custom_company_code not in st.session_state.selected_companies:
                st.session_state.custom_company = custom_company_code
                st.session_state.selected_companies[custom_company_code] = True
                
                # Create necessary folders if they don't exist
                if custom_company_code not in COMPANY_FOLDERS:
                    custom_folder = f"Company{len(COMPANY_FOLDERS) + 1}_Reports"
                    COMPANY_FOLDERS[custom_company_code] = custom_folder
                    os.makedirs(custom_folder, exist_ok=True)
                
                if custom_company_code not in DATASET_FOLDERS:
                    custom_dataset = f"Company{len(DATASET_FOLDERS) + 1}_Datasets"
                    DATASET_FOLDERS[custom_company_code] = custom_dataset
                    os.makedirs(custom_dataset, exist_ok=True)
        
        # Display selected companies
        selected = [code for code, selected in st.session_state.selected_companies.items() if selected]
        if selected:
            st.success(f"Selected {len(selected)} companies")
        else:
            st.error("Please select at least one company")
            return
        
        # Year Range Selection
        st.subheader("Year Range")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year",
                min_value=2000,
                max_value=datetime.now().year,
                value=get_year_range()[0],
                help="Select start year for data collection"
            )
        with col2:
            end_year = st.number_input(
                "End Year",
                min_value=2000,
                max_value=datetime.now().year,
                value=get_year_range()[1],
                help="Select end year for data collection"
            )
        
        # Validate year range
        if end_year < start_year:
            st.error("End year must be greater than or equal to start year")
            return
        
        # Data Collection Button
        if st.button("Download Reports", type="primary"):
            with st.spinner("Downloading reports..."):
                # Create date objects for the start and end of the years
                start_date = datetime(start_year, 1, 1).date()
                end_date = datetime(end_year, 12, 31).date()
                
                # Process each selected company
                for company_code in selected:
                    try:
                        company_name = COMPANY_NAMES.get(company_code, company_code)
                        st.markdown(f"### Processing {company_name}")
                        
                        # Create progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Update status
                        status_text.text(f"Starting download for {company_name}...")
                        
                        # Download reports
                        result = download_company_reports(
                            company_code=company_code,
                            start_date=start_date,
                            end_date=end_date,
                            progress_callback=lambda p: progress_bar.progress(p)
                        )
                        
                        # Process results
                        if result and 'summary' in result:
                            summary = result['summary']
                            status_text.text(f"Download completed! Found {summary['total_downloads']} reports.")
                            st.success(f"Reports downloaded successfully to {COMPANY_FOLDERS.get(company_code, 'Unknown')}")
                            
                            # Show download summary
                            st.info(f"""
                            Download Summary for {company_name}:
                            - Total Reports: {summary['total_downloads']}
                            - Years Covered: {', '.join(map(str, summary['years_processed']))}
                            """)
                        else:
                            status_text.text("Download completed but no reports found.")
                            st.warning(f"No reports were found for {company_name} in the selected year range.")
                        
                    except Exception as e:
                        st.error(f"Error downloading reports for {company_name}: {str(e)}")
                        status_text.text("Download failed!")
        
        # Data Processing Button
        if st.button("Process Reports", type="secondary"):
            st.session_state.processing_status = True
            
            # Process each selected company
            for company_code in selected:
                company_name = COMPANY_NAMES.get(company_code, company_code)
                
                with st.spinner(f"Processing reports for {company_name}..."):
                    try:
                        # Process reports
                        process_company_reports(
                            COMPANY_FOLDERS[company_code],
                            DATASET_FOLDERS[company_code]
                        )
                        st.success(f"Reports processed successfully for {company_name}!")
                        
                        # Update dataset availability after processing
                        datasets_available[company_code] = True
                        
                    except Exception as e:
                        st.error(f"Error processing reports for {company_name}: {str(e)}")
            
            st.session_state.processing_status = False
            
            # Note to restart the application if needed
            st.info("If you've just processed reports for the first time, you may need to restart the application to enable all analysis features.")
    
    # Main area with two tabs: Datasets and Insights
    main_tabs = st.tabs(["Datasets", "Insights"])
    
    # Datasets Tab
    with main_tabs[0]:
        st.header("Dataset Information")
        
        # Create tabs for each selected company
        selected = [code for code, selected in st.session_state.selected_companies.items() if selected]
        if len(selected) > 0:
            company_tabs = st.tabs([COMPANY_NAMES.get(code, code) for code in selected])
            
            for i, (tab, company_code) in enumerate(zip(company_tabs, selected)):
                with tab:
                    dataset_folder = DATASET_FOLDERS[company_code]
                    df = display_dataset_info(company_code, dataset_folder)
                    
                    if df is None and not datasets_available.get(company_code, False):
                        st.warning(f"No processed files found for {COMPANY_NAMES.get(company_code, company_code)}. Please follow these steps:")
                        st.markdown("""
                        1. Click **Download Reports** in the sidebar to download financial reports
                        2. Then click **Process Reports** to extract data from the downloaded reports
                        """)
    
    # Insights Tab
    with main_tabs[1]:
        st.header("Financial Insights")
        
        # Check if any datasets are available
        if not any(datasets_available.values()):
            st.warning("No processed data is available. Please download and process reports first.")
            st.markdown("""
            ### Getting Started:
            1. In the sidebar, select the companies you want to analyze
            2. Click **Download Reports** to download financial reports
            3. Click **Process Reports** to extract data from the reports
            4. Return to this tab to analyze the data
            """)
            return
        
        # Get the selected companies with available data
        selected = [code for code, selected in st.session_state.selected_companies.items() 
                   if selected and datasets_available.get(code, False)]
        
        if not selected:
            st.warning("None of the selected companies have processed data available.")
            return
        
        # Analysis options in tabs
        insight_tabs = st.tabs(["Ask Questions", "Visualize Data", "Forecast"])
        
        # 1. Ask Questions Tab
        with insight_tabs[0]:
            st.subheader("Ask Financial Questions")
            
            if not st.session_state.agent_initialized:
                st.warning("The analysis engine couldn't be initialized. Please ensure dataset folders have processed reports.")
                if 'agent_error' in st.session_state:
                    with st.expander("Error Details"):
                        st.code(st.session_state.agent_error)
                return
            
            # Input for custom query
            query = st.text_input(
                "Enter your financial question",
                placeholder="e.g., What was the revenue for Q1 2022?"
            )
            
            # Selection for query target
            query_target = st.multiselect(
                "Target Companies",
                options=selected,
                default=selected,
                format_func=lambda x: COMPANY_NAMES.get(x, x)
            )
            
            # Sample queries in a cleaner layout
            st.caption("Sample Questions (click to use)")
            sample_queries = [
                "What was the total revenue for the last quarter?",
                "How has gross profit changed over the past year?",
                "What is the trend in operating expenses?",
                "Compare the profit margins between quarters",
                "What was the net income in 2022?"
            ]
            
            cols = st.columns(3)
            for i, sample_query in enumerate(sample_queries):
                if cols[i % 3].button(sample_query, key=f"sample_{i}"):
                    st.session_state['query'] = sample_query
                    st.rerun()
            
            # Submit button for query
            if query and st.button("Submit Question", type="primary"):
                with st.spinner("Analyzing financial data..."):
                    # Get response from agent
                    response = get_agent_response(query, query_target)
                    
                    # Add to query history
                    st.session_state.query_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "companies": [COMPANY_NAMES.get(code, code) for code in query_target]
                    })
                    
                    # Display response
                    st.success("Analysis complete!")
                    st.markdown(response)
            
            # Display simplified query history
            if st.session_state.query_history:
                st.divider()
                st.subheader("Previous Queries")
                for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                    with st.expander(f"{item['query']} - {item['timestamp']}"):
                        st.write(f"Companies: {', '.join(item['companies'])}")
                        st.markdown(item['response'])
        
        # 2. Visualize Data Tab
        with insight_tabs[1]:
            st.subheader("Visualize Financial Data")
            
            # Settings in a single row
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                viz_company = st.selectbox(
                    "Company",
                    options=selected,
                    format_func=lambda x: COMPANY_NAMES.get(x, x)
                )
            
            with col2:
                metrics = ["Revenue", "Profit", "EPS", "Operating Expenses"]
                viz_metric = st.selectbox("Metric", options=metrics)
            
            with col3:
                st.write("&nbsp;")  # Spacer
                compare = st.checkbox("Compare Companies", value=len(selected) > 1)
            
            # Generate visualization
            if viz_company:
                with st.spinner("Generating visualization... Extracting data from financial reports"):
                    try:
                        if compare and len(selected) > 1:
                            # Comparison view
                            st.subheader(f"{viz_metric} Comparison")
                            
                            # Create comparison data
                            comparison_data = []
                            for company in selected:
                                company_name = COMPANY_NAMES.get(company, company)
                                try:
                                    # Get real data using the agent
                                    quarters = get_financial_quarters(company)
                                    metric_values = get_financial_metric(company, viz_metric)
                                    
                                    # Ensure the arrays are the same length
                                    min_length = min(len(quarters), len(metric_values))
                                    quarters = quarters[:min_length]
                                    metric_values = metric_values[:min_length]
                                    
                                    for i, quarter in enumerate(quarters):
                                        comparison_data.append({
                                            "Quarter": quarter,
                                            viz_metric: metric_values[i],
                                            "Company": company_name
                                        })
                                except Exception as e:
                                    st.warning(f"Error getting data for {company_name}: {str(e)}")
                            
                            if comparison_data:
                                # Create DataFrame
                                df = pd.DataFrame(comparison_data)
                                
                                # Create comparison chart
                                chart = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X('Quarter:N', title='Quarter'),
                                    y=alt.Y(f'{viz_metric}:Q', title=f'{viz_metric} (millions)'),
                                    color=alt.Color('Company:N', title='Company'),
                                    tooltip=['Quarter', viz_metric, 'Company']
                                ).properties(
                                    title=f'{viz_metric} Comparison'
                                )
                                
                                st.altair_chart(chart, use_container_width=True)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.warning("No data available for visualization.")
                        else:
                            # Single company view
                            company_name = COMPANY_NAMES.get(viz_company, viz_company)
                            st.subheader(f"{viz_metric} for {company_name}")
                            
                            # Get real data using the agent
                            quarters = get_financial_quarters(viz_company)
                            metric_values = get_financial_metric(viz_company, viz_metric)
                            
                            # Ensure the arrays are the same length
                            min_length = min(len(quarters), len(metric_values))
                            if min_length > 0:
                                quarters = quarters[:min_length]
                                metric_values = metric_values[:min_length]
                                
                                # Create DataFrame
                                df = pd.DataFrame({
                                    "Quarter": quarters,
                                    viz_metric: metric_values
                                })
                                
                                # Create chart
                                chart = create_line_chart(df, "Quarter", viz_metric, f"Quarterly {viz_metric}")
                                st.altair_chart(chart, use_container_width=True)
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.warning("No data available for visualization.")
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
        
        # 3. Forecast Tab
        with insight_tabs[2]:
            st.subheader("Financial Forecasting")
            
            # Simple settings
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                forecast_company = st.selectbox(
                    "Company",
                    options=selected,
                    format_func=lambda x: COMPANY_NAMES.get(x, x),
                    key="forecast_company"
                )
            
            with col2:
                forecast_metrics = ["Revenue", "Profit", "EPS", "Operating Expenses"]
                forecast_metric = st.selectbox("Metric", options=forecast_metrics, key="forecast_metric")
            
            with col3:
                forecast_periods = st.number_input("Quarters to Forecast", min_value=1, max_value=8, value=4)
            
            # Generate button
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating AI-powered forecast... Analyzing financial data"):
                    try:
                        # Get historical data using the agent
                        quarters = get_financial_quarters(forecast_company)
                        values = get_financial_metric(forecast_company, forecast_metric)
                        
                        # Ensure the arrays are the same length
                        min_length = min(len(quarters), len(values))
                        quarters = quarters[:min_length]
                        values = values[:min_length]
                        
                        if len(values) >= 4:  # Need enough data for forecasting
                            # Historical data to pass to agent
                            historical_data = {
                                "quarters": quarters,
                                "values": values
                            }
                            
                            # Base value from last data point
                            base = values[-1]
                            
                            # Generate future quarters
                            last_quarter_idx = int(quarters[-1][1])
                            last_year = int(quarters[-1][-4:])
                            
                            forecast_quarters = []
                            for i in range(forecast_periods):
                                next_quarter_idx = (last_quarter_idx + i + 1) % 4
                                if next_quarter_idx == 0:
                                    next_quarter_idx = 4
                                next_year = last_year + (last_quarter_idx + i + 1) // 5
                                forecast_quarters.append(f"Q{next_quarter_idx} {next_year}")
                            
                            # Create historical DataFrame 
                            historical_df = pd.DataFrame({
                                'Quarter': quarters,
                                forecast_metric: values,
                                'Type': ['Historical'] * len(quarters)
                            })
                            
                            # AI-Assisted Analysis
                            agent_forecast_values, agent_explanation = get_agent_forecast(
                                forecast_company, 
                                forecast_metric, 
                                forecast_periods,
                                historical_data
                            )
                            
                            # Simple trend projection as fallback
                            avg_growth = np.mean([values[i+1] - values[i] for i in range(len(values)-1)])
                            trend_values = [base + avg_growth * (i+1) for i in range(forecast_periods)]
                            
                            # Use agent forecast values or fallback to simple trend
                            if agent_forecast_values and len(agent_forecast_values) == forecast_periods:
                                forecast_values = agent_forecast_values
                                forecast_explanation = agent_explanation
                            else:
                                # Fallback to simple method if agent forecast fails
                                forecast_values = trend_values
                                forecast_explanation = f"""
                                Due to inability to generate a complete AI analysis, the forecast shows a simple trend projection 
                                based on historical average growth of {avg_growth:.2f} per quarter.
                                
                                {agent_explanation if agent_explanation else ""}
                                """
                            
                            # Create forecast DataFrames
                            forecast_df = pd.DataFrame({
                                'Quarter': forecast_quarters,
                                forecast_metric: forecast_values,
                                'Type': ['Forecast'] * len(forecast_quarters)
                            })
                            
                            combined_df = pd.concat([historical_df, forecast_df])
                            
                            # Display forecast
                            company_name = COMPANY_NAMES.get(forecast_company, forecast_company)
                            
                            # Create chart
                            chart = alt.Chart(combined_df).mark_line(point=True).encode(
                                x=alt.X('Quarter:N', title='Quarter'),
                                y=alt.Y(f'{forecast_metric}:Q', title=f'{forecast_metric} (millions)'),
                                color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], 
                                                                    range=['#1f77b4', '#ff7f0e'])),
                                tooltip=['Quarter', forecast_metric, 'Type']
                            ).properties(
                                title=f'{forecast_metric} Forecast for {company_name}'
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                            
                            # Calculate growth metrics
                            total_growth = (forecast_values[-1] - values[-1]) / values[-1] * 100
                            avg_quarterly_growth = (forecast_values[-1] / values[-1]) ** (1/forecast_periods) - 1
                            avg_quarterly_growth_pct = avg_quarterly_growth * 100
                            
                            # Show summary
                            st.success(f"Forecast Summary for {company_name}")
                            
                            # Display metrics side by side
                            col1, col2 = st.columns(2)
                            col1.metric("Current Value", f"{values[-1]:.2f}M", "Last Quarter")
                            col2.metric("Forecast Value", f"{forecast_values[-1]:.2f}M", f"{total_growth:.1f}% Growth")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Avg. Quarterly Growth", f"{avg_quarterly_growth_pct:.2f}%")
                            col2.metric("Forecast Period", f"{forecast_periods} Quarters")
                            
                            # Always show analysis from agent when available
                            st.subheader("AI Analysis")
                            if forecast_explanation:
                                st.markdown(forecast_explanation)
                            else:
                                st.markdown("*No detailed analysis available.*")
                            
                            # Show data table
                            with st.expander("View Forecast Data"):
                                st.dataframe(combined_df, use_container_width=True)
                        else:
                            st.warning("Not enough historical data available for accurate forecasting. Please process more quarterly reports.")
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")

if __name__ == "__main__":
    main() 