import streamlit as st
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import altair as alt
from parser import process_company_reports, COMPANY_FOLDERS, DATASET_FOLDERS, process_reports_from_ui, scan_company_reports, ensure_metadata_file
from scrapper import download_company_reports, download_reports
import json
from forecaster_agent import (
    time_series_forecast, 
    moving_average_forecast, 
    exponential_smoothing_forecast, 
    arima_forecast, 
    get_metric_timeseries,
    run_arima_forecast,
    extract_metric_from_question,
    run_forecast_from_question,
    plot_forecast,
    METRIC_MAPPINGS,
    get_financial_data
)
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time


# Set page config
st.set_page_config(
    page_title="Quarterly Financial Report Analyzer",
    page_icon="📊",
    layout="wide"
)

def main():
    # Add the title at the top of the sidebar
    st.sidebar.title("Financial Analysis")
    st.sidebar.caption("by Kevin Gomez")
    
    # Check workflow states
    reports_downloaded = check_reports_downloaded()
    datasets_available = check_datasets_available()
    
    # Determine if any reports are downloaded or datasets processed
    any_reports_downloaded = any(reports_downloaded.values())
    any_datasets_available = any(datasets_available.values())
    
    # Navigation options based on workflow state
    available_pages = ["Home", "Download Reports"]
    
    if any_reports_downloaded:
        available_pages.append("Process Reports")
        
    if any_datasets_available:
        available_pages.extend(["Data Visualization", "Financial Forecast"])
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "",  # No label needed as we have the markdown header
        available_pages
    )
    
    # Display workflow status
    with st.sidebar.expander("Workflow Status"):
        for company_code, company_name in COMPANY_NAMES.items():
            st.markdown(f"**{company_name}:**")
            
            # Report status
            if reports_downloaded.get(company_code, False):
                st.success("Reports Downloaded ✓")
            else:
                st.error("Reports Not Downloaded ✗")
            
            # Dataset status
            if datasets_available.get(company_code, False):
                st.success("Data Processed ✓")
            else:
                st.error("Data Not Processed ✗")
    
    # Display page based on selection
    if page == "Home":
        home_page()
    elif page == "Download Reports":
        download_reports_page()
    elif page == "Process Reports":
        process_reports_page(datasets_available)
    elif page == "Data Visualization":
        data_visualization_page()
    elif page == "Financial Forecast":
        financial_forecast_page()

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

def get_year_range():
    """Get default year range (last three years)"""
    current_year = datetime.now().year
    # Return the last three years (not including current year)
    # For example, if current year is 2025, return (2022, 2024)
    return current_year - 3, current_year - 1

def get_available_report_years(company_code):
    """Get available report years from metadata"""
    try:
        # Ensure metadata file exists
        ensure_metadata_file()
        
        # Try to read the metadata file
        with open("reports_metadata.json", "r") as f:
            metadata = json.load(f)
            
        if company_code in metadata:
            # Get current year for validation
            current_year = datetime.now().year
            # Filter years to only include last 5 years
            years = [year for year in metadata[company_code].get("available_years", [])
                    if year.isdigit() and current_year - 5 <= int(year) <= current_year]
            # Sort years in descending order (newest first)
            return sorted(years, reverse=True)
        return []
    except Exception as e:
        print(f"Error getting available years: {e}")
        return []

def get_available_quarters(company_code, year):
    """Get available quarters for reports based on metadata"""
    try:
        # Ensure metadata file exists
        ensure_metadata_file()
        
        # Try to read the metadata file
        with open("reports_metadata.json", "r") as f:
            metadata = json.load(f)
            
        if company_code in metadata and year in metadata[company_code].get("quarters_by_year", {}):
            return sorted(metadata[company_code]["quarters_by_year"][year])
        return []
    except Exception as e:
        print(f"Error getting available quarters: {e}")
        return []

def get_missing_quarters(company_code, year):
    """Get missing quarters for reports based on metadata"""
    try:
        # Ensure metadata file exists
        ensure_metadata_file()
        
        # Try to read the metadata file
        with open("reports_metadata.json", "r") as f:
            metadata = json.load(f)
            
        if company_code in metadata and year in metadata[company_code].get("missing_quarters", {}):
            return sorted(metadata[company_code]["missing_quarters"][year])
        return ["Q1", "Q2", "Q3", "Q4"]  # Default to all quarters missing
    except Exception as e:
        print(f"Error getting missing quarters: {e}")
        return ["Q1", "Q2", "Q3", "Q4"]  # Default to all quarters missing

def get_fiscal_calendar(company_code):
    """Get fiscal calendar for a company"""
    try:
        # Ensure metadata file exists
        ensure_metadata_file()
        
        # Try to read the metadata file
        with open("reports_metadata.json", "r") as f:
            metadata = json.load(f)
            
        if company_code in metadata and "fiscal_calendar" in metadata[company_code]:
            return metadata[company_code]["fiscal_calendar"]
        return {
            "Q1": "Jun",
            "Q2": "Sep",
            "Q3": "Dec",
            "Q4": "Mar"
        }
    except Exception as e:
        print(f"Error getting fiscal calendar: {e}")
        return {
            "Q1": "Jun",
            "Q2": "Sep",
            "Q3": "Dec",
            "Q4": "Mar"
        }

def display_dataset_info(company_code, folder_path):
    """Display information about the dataset with options to view file content"""
    if not os.path.exists(folder_path):
        st.warning(f"Dataset folder {folder_path} does not exist. Please process reports first.")
        return None
        
    # Check for financial metrics files
    files = [f for f in os.listdir(folder_path) if f.endswith('.json') and not f.startswith('.')]
    
    # Check for audit logs
    audit_logs_path = os.path.join(folder_path, "audit_logs")
    has_audit_logs = os.path.exists(audit_logs_path)
    if has_audit_logs:
        audit_logs = [f for f in os.listdir(audit_logs_path) if f.endswith('.json') and not f.startswith('.')]
    else:
        audit_logs = []
    
    # Create tabs for financial metrics and audit logs
    if files or audit_logs:
        dataset_tabs = st.tabs(["Financial Metrics", "Audit Logs"])
        
        # Generate a unique key prefix for this instance
        unique_id = f"{company_code}_{int(datetime.now().timestamp())}"
        
        # Financial Metrics Tab
        with dataset_tabs[0]:
            if files:
                # Get file sizes and creation dates
                file_info = []
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    size = os.path.getsize(file_path)
                    date = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    # Try to extract quarter and year from the JSON
                    quarter = "Unknown"
                    year = "Unknown"
                    status = "Unknown"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            quarter = data.get('quarter', "Unknown")
                            year = data.get('year', "Unknown")
                            status = data.get('status', "Unknown")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                    
                    file_info.append({
                        'File': file,
                        'Quarter': quarter,
                        'Year': year,
                        'Status': status,
                        'Size (KB)': round(size/1024, 2),
                        'Created': date.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                # Create DataFrame
                df = pd.DataFrame(file_info)
                
                # Sort by year and quarter if possible
                try:
                    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                    df = df.sort_values(['Year', 'Quarter'], ascending=[False, False])
                except Exception:
                    pass
                
                # Show dataframe
                st.dataframe(df, use_container_width=True)
                
        # Audit Logs Tab
        with dataset_tabs[1]:
            if audit_logs:
                # Get file sizes and creation dates
                log_info = []
                for file in audit_logs:
                    file_path = os.path.join(audit_logs_path, file)
                    size = os.path.getsize(file_path)
                    date = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    # Try to extract quarter and year from the JSON
                    quarter = "Unknown"
                    year = "Unknown"
                    status = "Unknown"
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            quarter = data.get('quarter', "Unknown")
                            year = data.get('year', "Unknown")
                            status = data.get('status', "Unknown")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                    
                    log_info.append({
                        'File': file,
                        'Quarter': quarter,
                        'Year': year,
                        'Status': status,
                        'Size (KB)': round(size/1024, 2),
                        'Created': date.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                # Create DataFrame
                logs_df = pd.DataFrame(log_info)
                
                # Sort by year and quarter if possible
                try:
                    logs_df['Year'] = pd.to_numeric(logs_df['Year'], errors='coerce')
                    logs_df = logs_df.sort_values(['Year', 'Quarter'], ascending=[False, False])
                except Exception:
                    pass
                
                # Show dataframe
                st.dataframe(logs_df, use_container_width=True)
                
        return df if files else None
    else:
        st.warning(f"No files found in {folder_path}. Please process reports first.")
    
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
    """Get quarterly data points from the financial dataset"""
    try:
        # Get the dataset folder path
        dataset_folder = DATASET_FOLDERS.get(company_code)
        
        # Check if folder exists
        if dataset_folder is None or not os.path.exists(dataset_folder):
            return []
        
        # Get all JSON files in the dataset folder
        json_files = [f for f in os.listdir(dataset_folder) if f.endswith('.json')]
        
        if not json_files:
            return []
        
        # Load quarter data from all JSON files
        quarters = []
        for file_name in json_files:
            file_path = os.path.join(dataset_folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    quarter = data.get('quarter')
                    year = data.get('year')
                    if quarter and year and quarter not in quarters:
                        # Format as "Q1 2022"
                        quarters.append(f"{quarter} {year}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Sort quarters chronologically
        def quarter_sort_key(q):
            # Extract year and quarter number for sorting
            if ' ' in q:  # Format like "Q1 2022"
                quarter_part, year_part = q.split(' ', 1)
                try:
                    year = int(year_part)
                    quarter_num = int(quarter_part[1]) if (quarter_part.startswith('Q') and len(quarter_part) > 1 and quarter_part[1].isdigit()) else 0
                    return (year, quarter_num)
                except (ValueError, IndexError):
                    return (0, 0)
            elif q.startswith('Q') and len(q) > 1:  # Format like "Q2"
                try:
                    quarter_num = int(q[1]) if q[1].isdigit() else 0
                    return (0, quarter_num)  # No year, sort by quarter only
                except (ValueError, IndexError):
                    return (0, 0)
            return (0, 0)  # Default sorting value
        
        quarters.sort(key=quarter_sort_key)
        
        return quarters
    
    except Exception as e:
        print(f"Error getting quarters: {e}")
        return []

def get_financial_metric(company_code, metric_name):
    """Get financial metric values from the financial dataset"""
    try:
        # Get the dataset folder path
        dataset_folder = DATASET_FOLDERS.get(company_code)
        
        # Check if folder exists
        if dataset_folder is None or not os.path.exists(dataset_folder):
            return []
        
        # Get all JSON files in the dataset folder
        json_files = [f for f in os.listdir(dataset_folder) if f.endswith('.json')]
        
        if not json_files:
            return []
        
        # Map the metric name to the corresponding JSON field
        metric_field_map = {
            "Revenue": "revenue",
            "COGS": ["cogs", "cost_of_sales"],
            "Gross Profit": "gross_profit",
            "Operating Expenses": ["operating_expenses", "distribution_costs", "administrative_expenses"],
            "Operating Income": "operating_income",
            "Net Income": ["profit_for_period", "net_profit"],
            "Finance Income": "finance_income",
            "Finance Costs": "finance_costs",
            "Profit Before Tax": "profit_before_tax",
            "Income Tax": ["tax_expense", "income_tax"],
            "Basic EPS": ["eps", "basic_eps"]
        }
        
        # Get field names to look for
        field_names = metric_field_map.get(metric_name, [metric_name.lower().replace(" ", "_")])
        if not isinstance(field_names, list):
            field_names = [field_names]
        
        # Load metric data from all JSON files
        quarters_data = {}
        for file_name in json_files:
            file_path = os.path.join(dataset_folder, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    quarter = data.get('quarter')
                    year = data.get('year')
                    
                    # Try each possible field name
                    metric_value = None
                    for field in field_names:
                        if field in data and data[field] is not None:
                            metric_value = data[field]
                            break
                    
                    # If we have all required data
                    if quarter and year and metric_value is not None:
                        # Format as "Q1 2022"
                        formatted_quarter = f"{quarter} {year}"
                        
                        # Convert to float if it's a number
                        try:
                            metric_value = float(metric_value)
                            quarters_data[formatted_quarter] = metric_value
                        except (ValueError, TypeError):
                            # If not a number, skip
                            pass
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Get sorted quarters
        sorted_quarters = get_financial_quarters(company_code)
        
        # Create values list in the same order as quarters
        values = []
        for quarter in sorted_quarters:
            if quarter in quarters_data:
                values.append(quarters_data[quarter])
            else:
                # Use None for missing values
                values.append(None)
        
        return values
    
    except Exception as e:
        print(f"Error getting {metric_name}: {e}")
        return []

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
            files = [f for f in os.listdir(folder) if f.endswith('.json') and not f.startswith('.')]
            available_datasets[company_code] = len(files) > 0
        else:
            available_datasets[company_code] = False
    
    return available_datasets

def display_forecast_graph(forecast_result, company_code, metric_name):
    """
    Display the forecast graph with confidence intervals
    
    Args:
        forecast_result: Dictionary with forecast results
        company_code: Company ticker symbol
        metric_name: Metric name for display
    """
    if "error" in forecast_result:
        st.error(forecast_result["error"])
        return
    
    # Check if we have the data needed for plotting
    if not all(key in forecast_result for key in ["forecast", "historical_data"]):
        st.error("Forecast result is missing required data for plotting")
        return
    
    # Get company display name
    company_name = COMPANY_NAMES.get(company_code, company_code)
    
    # Get the metric display name
    for display_name, internal_name in METRIC_MAPPINGS.items():
        if internal_name == metric_name:
            metric_display = display_name
            break
    else:
        metric_display = metric_name.capitalize()
    
    # Check if we have confidence intervals
    has_confidence = all(key in forecast_result for key in ["lower_ci", "upper_ci"])
    
    # Create DataFrame for Altair
    historical_data = forecast_result["historical_data"]
    forecast_values = forecast_result["forecast"]
    
    # Create period labels
    period_labels = forecast_result.get("period_labels", [f"Period {i+1}" for i in range(len(historical_data))])
    
    # Create future period labels
    if period_labels:
        last_period = period_labels[-1]
        parts = last_period.split()
        
        if len(parts) >= 2:
            quarter, year = parts[0], parts[1]
            # Extract quarter number
            q_num = int(quarter[1])
            
            future_labels = []
            current_year = int(year)
            current_q = q_num
            
            for _ in range(len(forecast_values)):
                current_q += 1
                if current_q > 4:
                    current_q = 1
                    current_year += 1
                future_labels.append(f"Q{current_q} {current_year}")
        else:
            # Fallback if we can't parse the period format
            future_labels = [f"Future {i+1}" for i in range(len(forecast_values))]
    else:
        future_labels = [f"Future {i+1}" for i in range(len(forecast_values))]
    
    # Create DataFrame for plotting
    df_hist = pd.DataFrame({
        "Quarter": period_labels,
        "Value": historical_data,
        "Type": "Historical"
    })
    
    df_forecast = pd.DataFrame({
        "Quarter": future_labels,
        "Value": forecast_values,
        "Type": "Forecast"
    })
    
    if has_confidence:
        df_forecast["Lower CI"] = forecast_result["lower_ci"]
        df_forecast["Upper CI"] = forecast_result["upper_ci"]
    
    # Combine historical and forecast data
    df = pd.concat([df_hist, df_forecast])
    
    # Create a line chart for historical and forecast values
    line_chart = alt.Chart(df).encode(
        x=alt.X('Quarter:N', title='Quarter', axis=alt.Axis(labelAngle=45)),
        y=alt.Y('Value:Q', title=f'{metric_display} (LKR thousands)'),
        color=alt.Color('Type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], 
                                                 range=['#1f77b4', '#d62728']))
    )
    
    # Draw the lines and points
    lines = line_chart.mark_line()
    points = line_chart.mark_circle(size=60)
    
    # Create the confidence interval area if available
    if has_confidence:
        # Filter to forecast data only
        forecast_data = df[df['Type'] == 'Forecast'].reset_index(drop=True)
        
        # Create a dataframe for the area chart - fix the concatenate error
        quarters = forecast_data['Quarter'].tolist()
        # Create a list that goes forwards then backwards to create the polygon shape
        area_quarters = quarters + quarters[::-1]
        
        # Values for lower and upper bounds
        lower_bounds = forecast_data['Lower CI'].tolist()
        upper_bounds = forecast_data['Upper CI'].tolist()[::-1]
        area_values = lower_bounds + upper_bounds
        
        # Create order column for the polygon
        area_order = list(range(len(quarters))) + list(range(len(quarters)))[::-1]
        
        area_df = pd.DataFrame({
            'Quarter': area_quarters,
            'Value': area_values,
            'Order': area_order
        })
        
        # Create the area chart
        area = alt.Chart(area_df).mark_area(opacity=0.3, color='#d62728').encode(
            x=alt.X('Quarter:N', title='Quarter'),
            y=alt.Y('Value:Q'),
            order='Order:Q'
        )
        
        # Combine charts
        combined_chart = (area + lines + points).properties(
            width=700,
            height=400,
            title=f'{company_name} - {metric_display} Forecast'
        )
    else:
        # Just lines and points if no confidence intervals
        combined_chart = (lines + points).properties(
            width=700,
            height=400,
            title=f'{company_name} - {metric_display} Forecast'
        )
    
    # Display the chart
    st.altair_chart(combined_chart, use_container_width=True)

def display_data_table(forecast_result, company_code, metric_name):
    """
    Display the data table with historical and forecasted values
    
    Args:
        forecast_result: Dictionary with forecast results
        company_code: Company ticker symbol
        metric_name: Metric name for display
    """
    if "error" in forecast_result:
        return
    
    # Get the data
    historical_data = forecast_result["historical_data"]
    forecast_values = forecast_result["forecast"]
    
    # Get period labels
    period_labels = forecast_result.get("period_labels", [f"Period {i+1}" for i in range(len(historical_data))])
    
    # Create future period labels
    if period_labels:
        last_period = period_labels[-1]
        parts = last_period.split()
        
        if len(parts) >= 2:
            quarter, year = parts[0], parts[1]
            # Extract quarter number
            q_num = int(quarter[1])
            
            future_labels = []
            current_year = int(year)
            current_q = q_num
            
            for _ in range(len(forecast_values)):
                current_q += 1
                if current_q > 4:
                    current_q = 1
                    current_year += 1
                future_labels.append(f"Q{current_q} {current_year}")
        else:
            future_labels = [f"Future {i+1}" for i in range(len(forecast_values))]
    else:
        future_labels = [f"Future {i+1}" for i in range(len(forecast_values))]
    
    # Get the metric display name
    for display_name, internal_name in METRIC_MAPPINGS.items():
        if internal_name == metric_name:
            metric_display = display_name
            break
    else:
        metric_display = metric_name.capitalize()
    
    # Create DataFrames
    df_hist = pd.DataFrame({
        "Quarter": period_labels,
        f"{metric_display} (LKR thousands)": [f"{x:,.2f}" for x in historical_data],
        "Type": "Historical"
    })
    
    # Check if we have confidence intervals
    has_confidence = all(key in forecast_result for key in ["lower_ci", "upper_ci"])
    
    if has_confidence:
        df_forecast = pd.DataFrame({
            "Quarter": future_labels,
            f"{metric_display} (LKR thousands)": [f"{x:,.2f}" for x in forecast_values],
            "Lower 95% CI (LKR thousands)": [f"{x:,.2f}" for x in forecast_result["lower_ci"]],
            "Upper 95% CI (LKR thousands)": [f"{x:,.2f}" for x in forecast_result["upper_ci"]],
            "Type": "Forecast"
        })
    else:
        df_forecast = pd.DataFrame({
            "Quarter": future_labels,
            f"{metric_display} (LKR thousands)": [f"{x:,.2f}" for x in forecast_values],
            "Type": "Forecast"
        })
    
    # Display historical data
    st.subheader("Historical Data")
    st.dataframe(df_hist.drop("Type", axis=1), use_container_width=True)
    
    # Display forecast data
    st.subheader("Forecast Data")
    st.dataframe(df_forecast.drop("Type", axis=1), use_container_width=True)

def display_forecast_description(forecast_result, company_code, question):
    """
    Display the forecast description
    
    Args:
        forecast_result: Dictionary with forecast results
        company_code: Company ticker symbol
        question: Forecasting question
    """
    if "error" in forecast_result:
        return
    
    company_name = COMPANY_NAMES.get(company_code, company_code)
    
    # Get ARIMA model order if available
    model_order = forecast_result.get("model_order", None)
    
    st.subheader("Forecast Description")
    
    if model_order:
        st.markdown(f"**ARIMA Model Order:** p={model_order[0]}, d={model_order[1]}, q={model_order[2]}")
        
        # Explain the model order
        p, d, q = model_order
        
        explanations = []
        
        if p > 0:
            explanations.append(f"**p={p}**: Uses {p} lagged observation(s) in the autoregressive model")
        
        if d > 0:
            explanations.append(f"**d={d}**: Uses {d} differencing to make the time series stationary")
        
        if q > 0:
            explanations.append(f"**q={q}**: Uses {q} lagged forecast error(s) in the moving average model")
        
        st.markdown("### Model Parameter Explanation")
        for exp in explanations:
            st.markdown(exp)
        
        # Explain how parameters were chosen
        st.markdown("### Parameter Selection Method")
        st.markdown("""
        The model parameters were selected using the Akaike Information Criterion (AIC), which balances model fit and simplicity. 
        The algorithm tested various combinations of p, d, and q values and selected the one that minimized the AIC value.
        """)
        
        # Seasonality explanation
        st.markdown("### Seasonality Handling")
        st.markdown("""
        This forecast uses quarterly seasonality (period = 4) for sufficient historical data.
        For shorter data series, seasonality might not be explicitly modeled.
        """)
    else:
        st.markdown("""
        The forecast was generated using an ARIMA model, which captures temporal dependencies in the time series data.
        """)
    
    if "aic" in forecast_result:
        st.markdown(f"**AIC (Akaike Information Criterion):** {forecast_result['aic']:.2f}")
    
    if "bic" in forecast_result:
        st.markdown(f"**BIC (Bayesian Information Criterion):** {forecast_result['bic']:.2f}")

def display_forecast_assumptions():
    """
    Display the forecast assumptions
    """
    st.subheader("Forecast Assumptions")
    
    with st.expander("View Assumptions"):
        st.markdown("""
        1. **Business Continuity**: The forecast assumes no major structural break in the business model
        2. **Trend Continuation**: Historical trends are assumed to continue without significant disruption
        3. **Data Limitations**: Only financial data is used (no macroeconomic or external factors)
        4. **Fiscal Calendar**: Fiscal calendar is based on Q1 ending in June, Q4 in March
        5. **Quarterly Patterns**: Quarterly seasonality patterns (if detected) are assumed to repeat
        6. **Stationarity**: After appropriate differencing, the time series is assumed to be stationary
        """)

def check_reports_downloaded():
    """Check if any reports have been downloaded"""
    reports_available = {}
    
    for company_code in COMPANY_NAMES.keys():
        years = get_available_report_years(company_code)
        reports_available[company_code] = len(years) > 0
    
    return reports_available

def check_enough_data_for_forecast(company_code, metric):
    """Check if there's enough data for forecasting (at least 4 data points)"""
    try:
        # Get the raw data
        data = get_financial_data(company_code, metric.lower())
        # Need at least 4 data points for a reasonable forecast
        return len(data) >= 4
    except Exception:
        return False

def delete_company_data(company_code):
    """
    Delete all processed data for a company
    
    Args:
        company_code: The company code to delete data for
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the dataset folder
        dataset_folder = DATASET_FOLDERS.get(company_code)
        
        if not dataset_folder or not os.path.exists(dataset_folder):
            return False
            
        # Delete all files in the dataset folder
        for filename in os.listdir(dataset_folder):
            file_path = os.path.join(dataset_folder, filename)
            
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
                
        # Update the reports metadata
        try:
            # Load metadata
            with open("reports_metadata.json", "r") as f:
                metadata = json.load(f)
                
            if company_code in metadata:
                # Reset processed data fields
                if "processed_files" in metadata[company_code]:
                    metadata[company_code]["processed_files"] = []
                    
                # Save updated metadata
                with open("reports_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Error updating metadata: {e}")
        
        return True
    except Exception as e:
        print(f"Error deleting data: {e}")
        return False

def financial_forecast_page():
    """Financial Forecast Page"""
    st.header("Financial Forecast")
    
    # Check if datasets are available
    datasets_available = check_datasets_available()
    
    if not any(datasets_available.values()):
        st.warning("No processed datasets available. Please process reports first.")
        st.info("1. Go to 'Download Reports' to download quarterly reports")
        st.info("2. Process the reports in 'Process Reports' page")
        st.info("3. Return to this page to generate forecasts")
        return
    
    # Dropdown menu for selecting company and forecasting question
    col1, col2 = st.columns(2)
    with col1:
        # Only show companies with available datasets
        available_companies = [code for code, available in datasets_available.items() if available]
        if not available_companies:
            st.error("No companies with processed data available.")
            return
            
        company_code = st.selectbox(
            "Select Company",
            available_companies,
            format_func=lambda x: COMPANY_NAMES[x],
            key="forecast_company"
        )
    
    with col2:
        forecasting_question = st.selectbox(
            "Select Forecasting Question",
            [
                "Forecast Net Profit for the next 4 quarters",
                "Estimate Revenue Trend over the coming year",
                "Predict EPS movement for upcoming quarters"
            ],
            key="forecast_question"
        )
    
    # Get the metric being forecasted
    metric = extract_metric_from_question(forecasting_question)
    
    # Check if there's enough data for forecasting
    enough_data = check_enough_data_for_forecast(company_code, metric)
    
    if not enough_data:
        st.warning(f"Insufficient data for {COMPANY_NAMES[company_code]} {metric.replace('_', ' ').title()} forecast.")
        st.info("At least 4 quarters of data are required for accurate forecasting.")
        st.info("Please process more quarterly reports to enable forecasting.")
        return
    
    # Add a button to run the forecast
    if st.button("Run Forecast", key="run_forecast"):
        with st.spinner("Running ARIMA forecast..."):
            # Run the forecast
            forecast_result = run_forecast_from_question(
                company=company_code,
                question=forecasting_question,
                periods=4  # Always forecast 4 quarters
            )
            
            # Check if the forecast used sample data (would have less than 4 points)
            if "historical_data" in forecast_result and len(forecast_result["historical_data"]) < 4:
                st.warning("Insufficient real data for reliable forecasting. Please process more quarterly reports.")
                return
            
            # Store the result in session state with different keys than the widgets
            st.session_state.forecast_result = forecast_result
            st.session_state.forecast_metric = metric
            st.session_state.stored_company = company_code
            st.session_state.stored_question = forecasting_question
    
    # Display forecast results if available
    if hasattr(st.session_state, 'forecast_result') and st.session_state.forecast_result:
        forecast_result = st.session_state.forecast_result
        metric = st.session_state.forecast_metric
        company_code = st.session_state.stored_company
        question = st.session_state.stored_question
        
        if "error" in forecast_result:
            st.error(forecast_result["error"])
        else:
            # Display forecast graph
            st.subheader("Forecast Graph")
            display_forecast_graph(forecast_result, company_code, metric)
            
            # Display data table
            display_data_table(forecast_result, company_code, metric)
            
            # Display forecast description
            display_forecast_description(forecast_result, company_code, question)
            
            # Display forecast assumptions
            display_forecast_assumptions()
    else:
        # Initial message when no forecast has been run
        st.info("Select a company and forecasting question, then click 'Run Forecast' to generate a forecast.")

def data_visualization_page():
    """Data Visualization Page"""
    st.header("Financial Metrics Visualization")
    
    # Check if datasets are available
    datasets_available = check_datasets_available()
    
    if not any(datasets_available.values()):
        st.warning("No processed datasets available. Please process reports first.")
        st.info("1. Go to 'Download Reports' to download quarterly reports")
        st.info("2. Process the reports in 'Process Reports' page")
        st.info("3. Return to this page to visualize financial metrics")
        return
    
    # Only show companies with available datasets
    available_companies = [code for code, available in datasets_available.items() if available]
    if not available_companies:
        st.error("No companies with processed data available.")
        return
        
    # Create tabs for different visualization options
    viz_tabs = st.tabs(["Overall Metric Comparison", "Financial Overview"])
    
    # Overall Metric Comparison Tab (formerly Multi-Metric Comparison)
    with viz_tabs[0]:
        col1, col2 = st.columns([1, 2])
        with col1:
            company_code = st.selectbox(
                "Select Company",
                available_companies,
                format_func=lambda x: COMPANY_NAMES[x],
                key="multi_viz_company"
            )
            
        with col2:
            selected_metrics = st.multiselect(
                "Select Metrics to Compare",
                ["Revenue", "COGS", "Gross Profit", "Operating Expenses", "Operating Income", "Net Income"],
                default=["Revenue", "Net Income"],
                key="multi_viz_metrics"
            )
        
        if selected_metrics:
            # Get data for all selected metrics
            quarters = get_financial_quarters(company_code)
            
            if not quarters:
                st.warning(f"No datasets processed for {COMPANY_NAMES.get(company_code, company_code)}. Please process reports first.")
            else:
                multi_df = pd.DataFrame({'Quarter': quarters})
                
                for metric in selected_metrics:
                    values = get_financial_metric(company_code, metric)
                    multi_df[metric] = values
                
                # Create and display multiple metric charts
                create_multi_metric_chart(multi_df, selected_metrics)
                
                # Display the data table
                with st.expander("View Data Table"):
                    st.dataframe(multi_df, use_container_width=True)
                
                # Download option
                offer_download(multi_df, f"multi_metric_comparison")
    
    # Financial Overview Tab
    with viz_tabs[1]:
        company_code = st.selectbox(
            "Select Company",
            available_companies,
            format_func=lambda x: COMPANY_NAMES[x],
            key="overview_company"
        )
        
        # Get all metrics for the company
        quarters = get_financial_quarters(company_code)
        
        if not quarters:
            st.warning(f"No datasets processed for {COMPANY_NAMES.get(company_code, company_code)}. Please process reports first.")
        else:
            metrics = ["Revenue", "COGS", "Gross Profit", "Operating Expenses", "Operating Income", "Net Income"]
            
            overview_data = {'Quarter': quarters}
            for metric in metrics:
                overview_data[metric] = get_financial_metric(company_code, metric)
            
            overview_df = pd.DataFrame(overview_data)
            
            # Special handling for operating expenses if missing
            if 'Operating Expenses' in overview_df.columns and overview_df['Operating Expenses'].isna().any():
                # Try to calculate from distribution_costs + administrative_expenses
                distr_costs = get_financial_metric(company_code, "Distribution Costs")
                admin_exp = get_financial_metric(company_code, "Administrative Expenses")
                
                if len(distr_costs) == len(admin_exp) == len(quarters):
                    # Calculate operating expenses as sum of distribution costs and administrative expenses
                    for i in range(len(quarters)):
                        if pd.isna(overview_df.loc[i, 'Operating Expenses']) and not pd.isna(distr_costs[i]) and not pd.isna(admin_exp[i]):
                            overview_df.loc[i, 'Operating Expenses'] = distr_costs[i] + admin_exp[i]
            
            # Special handling for Operating Income if missing
            if 'Operating Income' in overview_df.columns and overview_df['Operating Income'].isna().any():
                # Calculate operating income as gross profit - operating expenses
                for i in range(len(overview_df)):
                    # Only process if Operating Income is missing but the other values are present
                    if (pd.isna(overview_df.loc[i, 'Operating Income']) and 
                        not pd.isna(overview_df.loc[i, 'Gross Profit']) and 
                        not pd.isna(overview_df.loc[i, 'Operating Expenses'])):
                        
                        try:
                            # Use pandas to_numeric to safely convert to float
                            gross_profit = pd.to_numeric(overview_df.loc[i, 'Gross Profit'])
                            operating_expenses = pd.to_numeric(overview_df.loc[i, 'Operating Expenses'])
                            
                            # Calculate and store the result
                            overview_df.loc[i, 'Operating Income'] = gross_profit - operating_expenses
                        except Exception:
                            # Skip if conversion fails
                            pass
            
            # Calculate key ratios
            if len(overview_df) > 0:
                # Only calculate ratios for rows where both metrics are available
                overview_df['Gross Margin %'] = (overview_df['Gross Profit'] / overview_df['Revenue'] * 100).round(2)
                overview_df['Operating Margin %'] = (overview_df['Operating Income'] / overview_df['Revenue'] * 100).round(2)
                overview_df['Net Margin %'] = (overview_df['Net Income'] / overview_df['Revenue'] * 100).round(2)
            
            # Display summary metrics
            st.subheader("Key Performance Indicators")
            
            if len(overview_df) > 0:
                latest_quarter = overview_df.iloc[-1]
                
                # Only display metrics if they are available
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not pd.isna(latest_quarter['Revenue']) and not pd.isna(latest_quarter['Gross Margin %']):
                        st.metric(
                            "Latest Revenue",
                            f"Rs. {latest_quarter['Revenue']:,.0f}k",
                            f"{latest_quarter['Gross Margin %']:.1f}% margin"
                        )
                    else:
                        st.metric("Latest Revenue", "N/A")
                
                with col2:
                    if not pd.isna(latest_quarter['Operating Income']) and not pd.isna(latest_quarter['Operating Margin %']):
                        st.metric(
                            "Operating Income",
                            f"Rs. {latest_quarter['Operating Income']:,.0f}k",
                            f"{latest_quarter['Operating Margin %']:.1f}% margin"
                        )
                    else:
                        st.metric("Operating Income", "N/A")
                
                with col3:
                    if not pd.isna(latest_quarter['Net Income']) and not pd.isna(latest_quarter['Net Margin %']):
                        st.metric(
                            "Net Income",
                            f"Rs. {latest_quarter['Net Income']:,.0f}k",
                            f"{latest_quarter['Net Margin %']:.1f}% margin"
                        )
                    else:
                        st.metric("Net Income", "N/A")
            
            # Create tabs for different overview visualizations
            overview_viz_tabs = st.tabs(["Margins Analysis", "Quarterly Trends", "Full Data Table"])
            
            with overview_viz_tabs[0]:
                create_margins_chart(overview_df)
            
            with overview_viz_tabs[1]:
                create_quarterly_trends_chart(overview_df)
            
            with overview_viz_tabs[2]:
                # Create a copy of the dataframe to avoid modifying the original
                formatted_df = overview_df.copy()
                
                # Only format numeric columns
                for col in metrics + ['Gross Margin %', 'Operating Margin %', 'Net Margin %']:
                    if col in formatted_df.columns:
                        # Convert to numeric, coercing errors to NaN
                        formatted_df[col] = pd.to_numeric(formatted_df[col], errors='coerce')
                
                st.dataframe(formatted_df, use_container_width=True)
            
            # Download option
            offer_download(overview_df, "financial_overview")

def create_multi_metric_chart(df, metrics):
    """Create and display charts for multiple metrics"""
    # Check if dataframe is empty or has no data
    if df.empty or len(df) == 0 or all(df[metric].isna().all() for metric in metrics):
        st.warning("No data available to create chart.")
        return
        
    # Melt the dataframe for visualization
    df_melted = pd.melt(
        df,
        id_vars=['Quarter'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )
    
    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('Quarter:N', title='Quarter', sort=None),
        y=alt.Y('Value:Q', title='Value (Rs. thousands)'),
        color=alt.Color('Metric:N', title='Metric'),
        tooltip=['Quarter', 'Metric', 'Value']
    ).properties(
        width=700,
        height=400,
        title='Multiple Metrics Comparison'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    st.altair_chart(chart, use_container_width=True)
    
def create_margins_chart(df):
    """Create and display a chart for margin analysis"""
    margin_cols = ['Gross Margin %', 'Operating Margin %', 'Net Margin %']
    
    # Check if dataframe is empty or has no data
    if df.empty or len(df) == 0 or all(df[col].isna().all() for col in margin_cols if col in df.columns):
        st.warning("No margin data available to create chart.")
        return
        
    df_margins = pd.melt(
        df,
        id_vars=['Quarter'],
        value_vars=margin_cols,
        var_name='Margin Type',
        value_name='Percentage'
    )
    
    chart = alt.Chart(df_margins).mark_line(point=True).encode(
        x=alt.X('Quarter:N', title='Quarter', sort=None),
        y=alt.Y('Percentage:Q', title='Percentage (%)'),
        color=alt.Color('Margin Type:N', title='Margin Type'),
        tooltip=['Quarter', 'Margin Type', 'Percentage']
    ).properties(
        width=700,
        height=400,
        title='Margin Analysis Over Time'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    st.altair_chart(chart, use_container_width=True)

def create_quarterly_trends_chart(df):
    """Create and display a chart for quarterly trends"""
    metrics = ['Revenue', 'Operating Income', 'Net Income']
    
    # Check if dataframe is empty or has no data
    if df.empty or len(df) == 0 or all(df[metric].isna().all() for metric in metrics if metric in df.columns):
        st.warning("No quarterly trend data available to create chart.")
        return
        
    df_trends = pd.melt(
        df, 
        id_vars=['Quarter'], 
        value_vars=metrics,
        var_name='Metric',
        value_name='Value'
    )
    
    chart = alt.Chart(df_trends).mark_bar().encode(
        x=alt.X('Quarter:N', title='Quarter', sort=None),
        y=alt.Y('Value:Q', title='Value (Rs. thousands)'),
        color=alt.Color('Metric:N', title='Metric'),
        tooltip=['Quarter', 'Metric', 'Value']
    ).properties(
        width=700,
        height=400,
        title='Quarterly Performance Trends'
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    st.altair_chart(chart, use_container_width=True)
    
def offer_download(df, prefix):
    """Offer download options for the data"""
    col1, col2 = st.columns([1, 4])
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        st.markdown("*Download the data for further analysis*")

def download_reports_page():
    st.title("Download Quarterly Reports")
    
    # Select company to download reports for
    companies = st.multiselect(
        "Select Companies",
        ["REXP.N0000", "DIPD.N0000"],
        ["REXP.N0000"]  # Default selection
    )
    
    # Info message about automatic download
    st.info(
        "Reports will be automatically downloaded for the last three years "
        f"({datetime.now().year-3}, {datetime.now().year-2}, and {datetime.now().year-1}) "
        "based on the current date."
    )

    # Check if reports are already downloaded for selected companies
    all_reports_downloaded = True
    missing_reports = []
    
    for company in companies:
        years = get_available_report_years(company)
        if not years:
            all_reports_downloaded = False
            missing_reports.append(company)
            continue
            
        # Check if we have reports for the last three years
        current_year = datetime.now().year
        expected_years = {str(year) for year in [current_year-3, current_year-2, current_year-1]}
        available_years = set(years)
        
        if not expected_years.issubset(available_years):
            all_reports_downloaded = False
            missing_reports.append(company)
    
    # Only show download button if there are missing reports
    if not all_reports_downloaded and missing_reports:
        st.write("Reports need to be downloaded for:", ", ".join(missing_reports))
        if st.button("Download Reports"):
            # Start spinning animation
            with st.spinner(f"Downloading reports for {', '.join(companies)}..."):
                # Get current year for calculating the last three years
                current_year = datetime.now().year
                years_to_download = [current_year - 3, current_year - 2, current_year - 1]
                
                # Show what years are being used
                st.info(f"Downloading reports for years: {', '.join(map(str, years_to_download))}")
                    
                # Download reports for the selected companies
                try:
                    from scrapper import download_reports
                    result = download_reports(companies, years_to_download)
                    
                    if result and 'summary' in result:
                        st.success(f"Downloaded {result['summary']['total_downloads']} reports.")
                        
                        # After download, run enhanced quarter detection
                        st.info("Analyzing reports to detect quarters from content...")
                        for company in companies:
                            try:
                                from parser import scan_company_reports
                                with st.spinner(f"Running enhanced quarter detection for {company}..."):
                                    scan_results = scan_company_reports(company)
                                    if scan_results and isinstance(scan_results, dict) and scan_results.get("status") == "success":
                                        st.success(f"Quarter detection completed for {company}")
                                        
                                        # Display detailed results
                                        with st.expander(f"Quarter detection results for {company}"):
                                            st.write("**Years found:**", scan_results.get("years_found", []))
                                            st.write("**Available quarters by year:**")
                                            quarters_by_year = scan_results.get("quarters_by_year", {})
                                            if isinstance(quarters_by_year, dict):
                                                for year, quarters in quarters_by_year.items():
                                                    st.write(f"- {year}: {', '.join(quarters)}")
                                            st.write("**Missing quarters:**")
                                            missing_quarters = scan_results.get("missing_quarters", {})
                                            if isinstance(missing_quarters, dict):
                                                for year, quarters in missing_quarters.items():
                                                    st.write(f"- {year}: {', '.join(quarters) if quarters else 'None'}")
                            except Exception as e:
                                st.error(f"Error running quarter detection for {company}: {str(e)}")
                    else:
                        st.warning("No reports downloaded. Please check company selection and try again.")
                except Exception as e:
                    st.error(f"Error downloading reports: {str(e)}")
    else:
        if companies:
            st.success("All reports are already downloaded for the selected companies!")
        
    # Show available reports after download
    st.subheader("Available Reports")
    for company in companies:
        try:
            years = get_available_report_years(company)
            if years:
                st.write(f"**{company}**")
                for year in years:
                    quarters = get_available_quarters(company, year)
                    missing = get_missing_quarters(company, year)
                    
                    # Display available quarters
                    st.write(f"Year {year}: {', '.join(quarters)} (Missing: {', '.join(missing) if missing else 'None'})")
            else:
                st.write(f"**{company}**: No reports available")
        except Exception as e:
            st.error(f"Error loading report metadata for {company}: {str(e)}")

def home_page():
    """Home Page of the Application"""
    st.title("Quarterly Financial Report Analyzer")
    st.caption("by Kevin Gomez")
    
    st.markdown("""
    ## Welcome to the Quarterly Financial Report Analyzer
    
    This application helps you analyze financial reports for companies listed on the Colombo Stock Exchange.
    """)
    
    # Check workflow status
    reports_downloaded = check_reports_downloaded()
    datasets_available = check_datasets_available()
    
    # Show workflow steps with status
    st.subheader("Workflow Steps")
    
    # Step 1: Download Reports
    col1, col2 = st.columns([1, 3])
    with col1:
        if any(reports_downloaded.values()):
            st.success("1. Download Reports ✓")
        else:
            st.warning("1. Download Reports ⚠")
    with col2:
        if any(reports_downloaded.values()):
            st.markdown("Reports have been downloaded. You can download more in the 'Download Reports' page.")
        else:
            st.markdown("You need to download quarterly reports first. Go to the 'Download Reports' page.")
    
    # Step 2: Process Reports
    col1, col2 = st.columns([1, 3])
    with col1:
        if any(datasets_available.values()):
            st.success("2. Process Reports ✓")
        else:
            st.warning("2. Process Reports ⚠")
    with col2:
        if any(datasets_available.values()):
            st.markdown("Reports have been processed. You can process more in the 'Process Reports' page.")
        elif any(reports_downloaded.values()):
            st.markdown("You have downloaded reports but haven't processed them yet. Go to 'Process Reports'.")
        else:
            st.markdown("After downloading reports, you need to process them to extract financial data.")
    
    # Step 3: Visualize Data
    col1, col2 = st.columns([1, 3])
    with col1:
        if any(datasets_available.values()):
            st.success("3. Visualize Data ✓")
        else:
            st.warning("3. Visualize Data ⚠")
    with col2:
        if any(datasets_available.values()):
            st.markdown("You can visualize the processed data in the 'Data Visualization' page.")
        else:
            st.markdown("After processing reports, you can visualize financial metrics and trends.")
    
    # Step 4: Generate Forecasts
    col1, col2 = st.columns([1, 3])
    with col1:
        if any(datasets_available.values()):
            st.success("4. Generate Forecasts ✓")
        else:
            st.warning("4. Generate Forecasts ⚠")
    with col2:
        if any(datasets_available.values()):
            st.markdown("You can generate financial forecasts in the 'Financial Forecast' page.")
        else:
            st.markdown("After processing reports, you can generate forecasts for future quarters.")
    
    # Display company information and status
    st.subheader("Company Status")
    
    for company_code, company_name in COMPANY_NAMES.items():
        st.markdown(f"### {company_name} ({company_code})")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if reports_downloaded.get(company_code, False):
                st.success("Reports Downloaded")
                # Show number of available quarters/years
                years = get_available_report_years(company_code)
                st.write(f"{len(years)} years available")
            else:
                st.error("No Reports Downloaded")
                st.write("Go to 'Download Reports'")
        
        with col2:
            if datasets_available.get(company_code, False):
                st.success("Data Processed")
                # Show number of available quarters
                quarters = get_financial_quarters(company_code)
                st.write(f"{len(quarters)} quarters processed")
            else:
                st.error("No Data Processed")
                if reports_downloaded.get(company_code, False):
                    st.write("Go to 'Process Reports'")
                else:
                    st.write("First download reports")
        
        with col3:
            # Check if we have enough data for forecasting
            if datasets_available.get(company_code, False):
                # Check if we have at least one metric with enough data
                has_forecast_data = False
                for metric in ["revenue", "net_profit", "basic_eps"]:
                    if check_enough_data_for_forecast(company_code, metric):
                        has_forecast_data = True
                        break
                
                if has_forecast_data:
                    st.success("Ready for Forecasting")
                    st.write("Go to 'Financial Forecast'")
                else:
                    st.warning("Limited Data for Forecasting")
                    st.write("Need more quarters of data")
            else:
                st.error("Cannot Generate Forecasts")
                st.write("Process data first")

def process_reports_page(datasets_available):
    """Process Reports Page"""
    st.title("Process Quarterly Reports")
    
    # Create tabs for different operations - Create the tabs upfront
    tabs = st.tabs(["Process Reports", "Delete Processed Data"])
    
    # Check if reports are downloaded
    reports_downloaded = check_reports_downloaded()
    available_companies = []
    
    # Process Reports Tab
    with tabs[0]:
        st.header("Process New Reports")
        
        if not any(reports_downloaded.values()):
            st.warning("No reports have been downloaded yet. Please download reports first.")
            st.info("1. Go to 'Download Reports' to download quarterly reports")
            st.info("2. Return to this page to process the downloaded reports")
        else:
            # Only show companies that have reports downloaded
            available_companies = [code for code, available in reports_downloaded.items() if available]
            
            # Select companies to process
            selected_companies = []
            
            # Create columns for companies
            cols = st.columns(len(COMPANY_NAMES))
            
            for i, (company_code, company_name) in enumerate(COMPANY_NAMES.items()):
                with cols[i]:
                    # Only enable checkbox if reports are downloaded
                    disabled = company_code not in available_companies
                    
                    if disabled:
                        st.checkbox(
                            company_name, 
                            value=False, 
                            key=f"check_{company_code}",
                            disabled=True,
                            help="No reports downloaded for this company"
                        )
                        st.session_state.selected_companies[company_code] = False
                    else:
                        if st.checkbox(
                            company_name, 
                            value=st.session_state.selected_companies.get(company_code, False), 
                            key=f"check_{company_code}"
                        ):
                            selected_companies.append(company_code)
                            st.session_state.selected_companies[company_code] = True
                        else:
                            st.session_state.selected_companies[company_code] = False
            
            if not selected_companies:
                st.warning("Please select at least one company to process.")
            else:
                # Year and Quarter selection
                st.subheader("Select Reports to Process")
                
                # Create columns for year and quarter selection
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get all available years from metadata for selected companies
                    all_years = set()
                    for company_code in selected_companies:
                        years = get_available_report_years(company_code)
                        all_years.update(years)
                    
                    # Convert to sorted list
                    all_years = sorted(list(all_years), reverse=True)
                    
                    # Add "All Years" option
                    years_options = ["All Years"] + all_years
                    selected_year = st.selectbox(
                        "Select Year",
                        years_options,
                        key="process_year"
                    )
                
                with col2:
                    # Get all quarters for the selected year(s)
                    all_quarters = set()
                    for company_code in selected_companies:
                        if selected_year == "All Years":
                            # Get all quarters from all years
                            for year in all_years:
                                quarters = get_available_quarters(company_code, year)
                                all_quarters.update(quarters)
                        else:
                            quarters = get_available_quarters(company_code, selected_year)
                            all_quarters.update(quarters)
                    
                    # Convert to sorted list
                    all_quarters = sorted(list(all_quarters))
                    
                    # Add "All Quarters" option
                    quarters_options = ["All Quarters"] + all_quarters
                    selected_quarter = st.selectbox(
                        "Select Quarter",
                        quarters_options,
                        key="process_quarter"
                    )
                
                # Display selected reports to process
                st.subheader("Reports to Process")
                for company_code in selected_companies:
                    st.write(f"**{COMPANY_NAMES.get(company_code, company_code)}**")
                    
                    if selected_year == "All Years":
                        process_years = all_years
                    else:
                        process_years = [selected_year]
                        
                    if selected_quarter == "All Quarters":
                        process_quarters = all_quarters
                    else:
                        process_quarters = [selected_quarter]
                    
                    # Show which reports will be processed
                    for year in process_years:
                        quarters_in_year = get_available_quarters(company_code, year)
                        quarters_to_process = [q for q in process_quarters if q in quarters_in_year]
                        if quarters_to_process:
                            st.write(f"Year {year}: {', '.join(quarters_to_process)}")
                
                # Process button
                if st.button("Process Reports"):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Processing status
                    status_text = st.empty()
                    
                    # Define callback function for progress updates
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        
                        # Update status text
                        if progress < 0.2:
                            status_text.text("Reading reports...")
                        elif progress < 0.8:
                            status_text.text("Extracting financial metrics...")
                        else:
                            status_text.text("Finalizing results...")
                    
                    try:
                        # Convert selections to parameters for processing
                        if selected_year == "All Years":
                            start_year = min(int(y) for y in all_years)
                            end_year = max(int(y) for y in all_years)
                        else:
                            start_year = int(selected_year)
                            end_year = int(selected_year)
                        
                        # Run processing with progress updates
                        results = process_reports_from_ui(
                            selected_companies, 
                            start_year, 
                            end_year, 
                            update_progress,
                            selected_quarter if selected_quarter != "All Quarters" else None
                        )
                        
                        # Processing complete
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        
                        # Display processing results
                        st.subheader("Processing Results")
                        
                        for company_code in selected_companies:
                            st.write(f"**{COMPANY_NAMES.get(company_code, company_code)}**")
                            
                            # Get company-specific results
                            company_results = {k: v for k, v in results.items() if k.startswith(company_code)}
                            
                            if company_results:
                                # Count successes and errors
                                successes = sum(1 for result in company_results.values() if result.get("status") == "success")
                                errors = sum(1 for result in company_results.values() if result.get("status") == "error")
                                
                                st.write(f"Successfully processed {successes} reports, {errors} failures")
                                
                                # Show dataset info
                                dataset_folder = DATASET_FOLDERS.get(company_code)
                                if dataset_folder:
                                    df = display_dataset_info(company_code, dataset_folder)
                            else:
                                st.warning(f"No results for {company_code}")
                        
                        # Set processing status in session state
                        st.session_state.processing_status = True
                        
                    except Exception as e:
                        # Error in processing
                        progress_bar.progress(1.0)
                        status_text.text("Error in processing!")
                        st.error(f"Error processing reports: {str(e)}")
    
    # Delete Processed Data Tab
    with tabs[1]:
        st.header("Delete Processed Data")
        st.warning("⚠️ **Warning**: Deleting processed data is irreversible. You will need to process the reports again.")
        
        # Show only companies with processed data
        companies_with_data = [code for code, available in datasets_available.items() if available]
        
        if not companies_with_data:
            st.info("No processed data available for any company.")
        else:
            # Create a select box for companies with data
            delete_company = st.selectbox(
                "Select Company to Delete Data",
                companies_with_data,
                format_func=lambda x: COMPANY_NAMES.get(x, x),
                key="delete_company"
            )
            
            # Show dataset info for the selected company
            st.subheader(f"Data for {COMPANY_NAMES.get(delete_company, delete_company)}")
            dataset_folder = DATASET_FOLDERS.get(delete_company)
            if dataset_folder and os.path.exists(dataset_folder):
                display_dataset_info(delete_company, dataset_folder)
                
                # Confirmation for deletion
                st.subheader("Confirm Deletion")
                confirm_text = st.text_input(
                    f"Type 'DELETE {delete_company}' to confirm deletion of all processed data for this company:",
                    key="confirm_delete"
                )
                
                if st.button("Delete All Processed Data", key="delete_button"):
                    if confirm_text == f"DELETE {delete_company}":
                        with st.spinner(f"Deleting data for {COMPANY_NAMES.get(delete_company, delete_company)}..."):
                            success = delete_company_data(delete_company)
                            if success:
                                st.success(f"All processed data for {COMPANY_NAMES.get(delete_company, delete_company)} has been deleted.")
                                
                                # Create a placeholder for the refresh message
                                refresh_placeholder = st.empty()
                                refresh_placeholder.info("The page will refresh in 3 seconds...")
                                
                                # Use st.cache to avoid error with multiple rerun calls
                                time.sleep(1)
                                refresh_placeholder.info("The page will refresh in 2 seconds...")
                                time.sleep(1)
                                refresh_placeholder.info("The page will refresh in 1 second...")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed to delete data for {COMPANY_NAMES.get(delete_company, delete_company)}.")
                    else:
                        st.error("Confirmation text does not match. Data was not deleted.")
            else:
                st.info(f"No processed data found for {COMPANY_NAMES.get(delete_company, delete_company)}.")
    
    # Display available datasets section
    st.subheader("Available Datasets")
    
    for company_code, available in datasets_available.items():
        if available:
            st.write(f"**{COMPANY_NAMES.get(company_code, company_code)}**")
            
            # Display dataset info if available
            dataset_folder = DATASET_FOLDERS.get(company_code)
            if dataset_folder:
                display_dataset_info(company_code, dataset_folder)

if __name__ == "__main__":
    main() 