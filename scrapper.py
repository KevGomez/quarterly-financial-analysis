import os
import re
import requests
import time
import json
import concurrent.futures
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Define constants for the two permanent company folders
COMPANY_FOLDERS = {
    "REXP.N0000": "Company1_Reports",
    "DIPD.N0000": "Company2_Reports",
    # Add any additional company mappings here
}

# Default folder for any company not in the mapping
DEFAULT_FOLDER = "Other_Company_Reports"

def download_pdf(pdf_url, folder):
    """
    Downloads a PDF from the provided URL and saves it to the specified folder.
    """
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Extract the file name from the URL or set a default name
        filename = pdf_url.split("/")[-1] or f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        file_path = os.path.join(folder, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")
        return file_path
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return None

def download_pdf_parallel(pdf_url, folder):
    """
    Function wrapper for concurrent download
    """
    return {
        "url": pdf_url,
        "result": download_pdf(pdf_url, folder)
    }

def filter_quarterly_links(pdf_links_data, valid_years):
    """
    Filter PDF links to get quarterly reports for the specified years
    """
    if not pdf_links_data:
        return []
    
    # Create a list to store filtered links
    filtered_links = []
    
    # Count of reports per year to ensure we get all quarterly reports
    reports_per_year = {year: 0 for year in valid_years}
    report_details = {year: [] for year in valid_years}
    
    # Convert valid_years to strings for easier matching
    valid_years_str = [str(year) for year in valid_years]
    
    # Sort links by date (if possible) to get newest first
    sorted_links = sorted(
        pdf_links_data, 
        key=lambda x: x.get("text", ""), 
        reverse=True
    )
    
    print(f"\nAnalyzing {len(sorted_links)} potential quarterly reports...")
    
    # First pass: Find definite quarterly reports for each year
    print("\nIdentifying quarterly reports by year...")
    for link in sorted_links:
        text = link["text"].lower()
        
        # Check if this is a quarterly report
        is_quarterly = any(term in text for term in ["quarter", "interim", "q1", "q2", "q3", "q4"])
        
        # Check if text contains a valid year
        matching_year = next((year for year in valid_years_str if year in text), None)
        
        # Try to identify which quarter this is
        quarter = None
        if "q1" in text or "first quarter" in text or "1st quarter" in text or "march" in text or "mar" in text:
            quarter = "Q1"
        elif "q2" in text or "second quarter" in text or "2nd quarter" in text or "june" in text or "jun" in text:
            quarter = "Q2"
        elif "q3" in text or "third quarter" in text or "3rd quarter" in text or "september" in text or "sep" in text:
            quarter = "Q3"
        elif "q4" in text or "fourth quarter" in text or "4th quarter" in text or "december" in text or "dec" in text:
            quarter = "Q4"
        
        if is_quarterly and matching_year:
            year = int(matching_year)
            if year in valid_years and reports_per_year[year] < 4:
                if quarter:
                    # Check if we already have this quarter for this year
                    if any(q == quarter for q, _ in report_details[year]):
                        continue
                    
                filtered_links.append(link["href"])
                reports_per_year[year] += 1
                report_details[year].append((quarter or f"Unknown_{reports_per_year[year]}", link["text"]))
    
    # Second pass: If we're missing reports, use date-based heuristics
    for year in valid_years:
        if reports_per_year[year] < 4:
            # Check which quarters we already have
            existing_quarters = [q for q, _ in report_details[year]]
            missing_quarters = [f"Q{i}" for i in range(1, 5) if f"Q{i}" not in existing_quarters]
            
            # Look for links that might be quarterly reports for this year
            for link in sorted_links:
                if link["href"] not in filtered_links:
                    text = link["text"].lower()
                    
                    # If this link has the year, add it
                    if str(year) in text:
                        # Try to identify which quarter this might be
                        potential_quarter = None
                        if "q1" in text or "first" in text or "march" in text or "mar" in text:
                            potential_quarter = "Q1"
                        elif "q2" in text or "second" in text or "june" in text or "jun" in text:
                            potential_quarter = "Q2"
                        elif "q3" in text or "third" in text or "september" in text or "sep" in text:
                            potential_quarter = "Q3"
                        elif "q4" in text or "fourth" in text or "december" in text or "dec" in text:
                            potential_quarter = "Q4"
                        
                        # Add if it's a quarter we need
                        if potential_quarter in missing_quarters:
                            filtered_links.append(link["href"])
                            reports_per_year[year] += 1
                            report_details[year].append((potential_quarter, link["text"]))
                            missing_quarters.remove(potential_quarter)
                            
                            # Stop if we have all 4 reports for this year
                            if reports_per_year[year] >= 4:
                                break
                        elif potential_quarter is None and reports_per_year[year] < 4:
                            # If we can't determine the quarter but need reports, add it anyway
                            potential_quarter = f"Unknown_{reports_per_year[year] + 1}"
                            filtered_links.append(link["href"])
                            reports_per_year[year] += 1
                            report_details[year].append((potential_quarter, link["text"]))
                            
                            # Stop if we have all 4 reports for this year
                            if reports_per_year[year] >= 4:
                                break
    
    # Print detailed report count per year
    print("\nQuarterly Reports Summary:")
    total_reports = 0
    for year in sorted(valid_years):
        print(f"  {year}: {reports_per_year[year]}/4 reports")
        total_reports += reports_per_year[year]
        for quarter, text in sorted(report_details[year]):
            print(f"    - {quarter}: {text}")
    
    print(f"\nTotal reports found: {total_reports}/{len(valid_years)*4}")
    
    return filtered_links

def scrape_quarterly_reports(company_symbol, folder, valid_years):
    """
    Pipeline to scrape quarterly reports from the CSE website for a specific company.
    """
    base_url = f"https://www.cse.lk/pages/company-profile/company-profile.component.html?symbol={company_symbol}"
    
    # Setup Selenium with Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    print(f"Starting scraping process for {company_symbol}...")
    
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(base_url)
        
        # Wait for page to load and click on the Financials tab
        financials_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Financials')]"))
        )
        financials_tab.click()
        
        print("Clicked on Financials tab")
        
        # Wait for tab content to load and click on Quarterly Reports
        quarterly_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
        )
        quarterly_tab.click()
        
        print("Clicked on Quarterly Reports tab")
        
        # Wait for report data to load
        time.sleep(5)
        
        # Extract PDF links and metadata from the page
        pdf_links_data = []
        
        try:
            # Find all PDF icons
            pdf_icons = driver.find_elements(By.CSS_SELECTOR, "i.fa-file-pdf-o")
            print(f"Found {len(pdf_icons)} PDF icons with class fa-file-pdf-o")
            
            if pdf_icons:
                # First, extract row information for all icons
                icon_info = []
                for i, icon in enumerate(pdf_icons):
                    try:
                        # Get the row information
                        row = icon.find_element(By.XPATH, "./ancestor::tr")
                        row_text = row.text if row else f"PDF Document {i+1}"
                        
                        # Get date and report text from columns if available
                        try:
                            cols = row.find_elements(By.TAG_NAME, "td")
                            if len(cols) >= 2:
                                date_text = cols[0].text.strip()
                                report_text = cols[1].text.strip()
                                row_text = f"{date_text} - {report_text}"
                        except:
                            pass
                        
                        # Get parent element for JavaScript handlers/attributes
                        parent = icon.find_element(By.XPATH, "./..") if icon else None
                        
                        # Store icon info for later processing
                        icon_info.append({
                            "icon": icon,
                            "text": row_text,
                            "parent": parent,
                            "index": i
                        })
                    except Exception as e:
                        print(f"Error getting info for icon {i+1}: {e}")
                
                # Method 1: Check for href attributes in parent elements
                for info in icon_info:
                    if info["parent"] and info["parent"].tag_name == 'a':
                        href = info["parent"].get_attribute('href')
                        if href:
                            pdf_links_data.append({
                                "text": info["text"],
                                "href": href
                            })
                
                # Method 2: Check for onclick handlers
                for info in icon_info:
                    if info["parent"]:
                        onclick = info["parent"].get_attribute('onclick')
                        if onclick:
                            # Try to extract URL from onclick handler
                            url_match = re.search(r"window\.open\(['\"](.*?)['\"]", onclick)
                            if url_match:
                                href = url_match.group(1)
                                pdf_links_data.append({
                                    "text": info["text"],
                                    "href": href
                                })
                
                # Method 3: Extract URLs by monitoring network traffic when clicking icons
                if len(pdf_links_data) < len(icon_info):
                    print("Extracting PDF URLs by clicking icons...")
                    
                    # Create a new window to capture PDF URLs
                    network_options = Options()
                    network_options.add_argument("--window-size=1200,800")
                    network_options.add_argument("--no-sandbox")
                    network_options.add_argument("--disable-dev-shm-usage")
                    
                    network_driver = webdriver.Chrome(options=network_options)
                    try:
                        # Navigate to the page
                        network_driver.get(base_url)
                        
                        # Click Financials tab
                        WebDriverWait(network_driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Financials')]"))
                        ).click()
                        
                        # Click Quarterly Reports tab
                        WebDriverWait(network_driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Quarterly Reports')]"))
                        ).click()
                        
                        # Wait for content to load
                        time.sleep(5)
                        
                        # Check which icons we still need URLs for
                        missing_icons = [info for info in icon_info 
                                         if not any(data["text"] == info["text"] for data in pdf_links_data)]
                        
                        # Find PDF icons in the new browser
                        net_pdf_icons = network_driver.find_elements(By.CSS_SELECTOR, "i.fa-file-pdf-o")
                        
                        # For each missing icon, click and extract URL
                        for info in missing_icons:
                            if info["index"] < len(net_pdf_icons):
                                try:
                                    # Click the icon
                                    icon_to_click = net_pdf_icons[info["index"]]
                                    
                                    # Store the current handles before clicking
                                    original_handles = network_driver.window_handles
                                    
                                    # Click the PDF icon
                                    icon_to_click.click()
                                    
                                    # Wait for new window/tab to open
                                    time.sleep(3)
                                    
                                    # Check if a new window was opened
                                    new_handles = network_driver.window_handles
                                    if len(new_handles) > len(original_handles):
                                        # Switch to the new window
                                        new_handle = [h for h in new_handles if h not in original_handles][0]
                                        network_driver.switch_to.window(new_handle)
                                        
                                        # Get the URL of the PDF
                                        pdf_url = network_driver.current_url
                                        if pdf_url and pdf_url != base_url:
                                            pdf_links_data.append({
                                                "text": info["text"],
                                                "href": pdf_url
                                            })
                                        
                                        # Close the new window and switch back
                                        network_driver.close()
                                        network_driver.switch_to.window(original_handles[0])
                                    
                                    # Wait a bit before the next click
                                    time.sleep(2)
                                    
                                except Exception as e:
                                    print(f"Error clicking icon: {e}")
                    finally:
                        network_driver.quit()
        except Exception as e:
            print(f"Error extracting PDF links: {e}")
        
        # Backup method: Parse the page with BeautifulSoup
        if len(pdf_links_data) == 0:
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            
            # Look for PDF icons with the specific class
            pdf_icons = soup.find_all('i', class_=lambda c: c and ('fa-file-pdf-o' in c))
            
            for icon in pdf_icons:
                # Try to find parent link or JS click handler
                parent = icon.parent
                while parent and parent.name != 'a' and parent.name != 'tr':
                    parent = parent.parent
                
                if parent and parent.name == 'a':
                    href = parent.get('href', '')
                    if not href:
                        onclick = parent.get('onclick', '')
                        if 'window.open' in onclick:
                            href = re.search(r"window\.open\(['\"](.*?)['\"]", onclick)
                            if href:
                                href = href.group(1)
                    
                    if href:
                        # Get row information for text
                        row = icon.find_parent('tr')
                        if row:
                            cols = row.find_all('td')
                            if len(cols) >= 2:
                                date_text = cols[0].get_text(strip=True)
                                report_text = cols[1].get_text(strip=True)
                                text = f"{date_text} - {report_text}"
                            else:
                                text = row.get_text(strip=True)
                        else:
                            text = parent.get_text(strip=True) or "PDF Document"
                        
                        # Check if we already have this href
                        if not any(item["href"] == href for item in pdf_links_data):
                            pdf_links_data.append({
                                "text": text,
                                "href": href
                            })
                
        print(f"Total PDF links/icons found: {len(pdf_links_data)}")
        
        if not pdf_links_data:
            print("No PDF links or icons found.")
            return []
        
        # Filter links to ensure we get all quarterly reports (4 per year for past 3 years)
        filtered_pdf_urls = filter_quarterly_links(pdf_links_data, valid_years)
        
        print(f"Using {len(filtered_pdf_urls)} PDF links for download")
        
        # Download all PDFs in parallel
        print("Starting parallel download of all PDFs...")
        downloaded_files = []
        
        # Create a thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit download tasks
            future_to_url = {
                executor.submit(download_pdf_parallel, pdf_url, folder): pdf_url 
                for pdf_url in filtered_pdf_urls if not (os.path.exists(pdf_url))
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data["result"]:
                        downloaded_files.append(data["result"])
                except Exception as exc:
                    print(f"Download for {url} generated an exception: {exc}")
        
        # Add any local files that were already available
        for pdf_url in filtered_pdf_urls:
            if os.path.exists(pdf_url) and pdf_url not in downloaded_files:
                downloaded_files.append(pdf_url)
        
        return downloaded_files
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        return []
    finally:
        driver.quit()

def run_quarterly_reports_pipeline(company_symbols, years=None):
    """
    Main pipeline function to orchestrate the scraping and downloading process.
    
    Args:
        company_symbols: List of company symbols to scrape
        years: List of specific years to scrape (defaults to last 3 years if not specified)
    
    Returns:
        Dictionary with company symbols as keys and lists of downloaded files as values,
        plus a 'summary' key with the total number of downloads.
    """
    # Use default years (last 3) if not specified
    if not years:
        current_year = datetime.now().year
        years = [current_year - i for i in range(3)]  # e.g., [2023, 2022, 2021]
    
    # If a single company is provided as a string, convert to a list
    if isinstance(company_symbols, str):
        company_symbols = [company_symbols]
    
    total_downloads = 0
    all_files = {}
    
    # Process each company
    for company_symbol in company_symbols:
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPANY: {company_symbol}")
        print(f"{'='*60}")
        
        # Use permanent folder based on company mapping
        if company_symbol in COMPANY_FOLDERS:
            folder = COMPANY_FOLDERS[company_symbol]
        else:
            folder = os.path.join(DEFAULT_FOLDER, company_symbol)
            
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        print(f"Pipeline started: Downloading quarterly reports for {company_symbol} for years: {years}")
        print(f"Saving reports to folder: {folder}")
        
        # Execute the scraping and downloading
        downloaded_files = scrape_quarterly_reports(company_symbol, folder, years)
        
        if downloaded_files:
            print(f"\nPipeline completed successfully for {company_symbol}. Downloaded {len(downloaded_files)}/{len(years)*4} quarterly reports.")
            print("Downloaded reports:")
            for file in downloaded_files:
                print(f"  - {os.path.basename(file)}")
                
            # Check if we have fewer than expected
            expected_count = len(years) * 4
            if len(downloaded_files) < expected_count:
                print(f"\nNote: Only {len(downloaded_files)} of the expected {expected_count} quarterly reports were found.")
                print("The company may not have published all quarterly reports for the requested years.")
                
            total_downloads += len(downloaded_files)
            all_files[company_symbol] = downloaded_files
        else:
            print(f"Pipeline completed for {company_symbol} but no files were downloaded.")
            all_files[company_symbol] = []
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Processed {len(company_symbols)} companies for years: {years}")
    print(f"Total reports downloaded: {total_downloads}")
    
    # Add summary to the result
    all_files['summary'] = {
        'total_downloads': total_downloads,
        'companies_processed': len(company_symbols),
        'years_processed': years
    }
    
    return all_files

# Example function that can be called from a Streamlit app
def download_reports(company_symbols, years=None):
    """
    Function to be called from a Streamlit app.
    
    Args:
        company_symbols: List of company symbols or single company symbol as string
        years: List of years to download reports for (optional)
        
    Returns:
        Dictionary with results of the download operation
    """
    return run_quarterly_reports_pipeline(company_symbols, years)

def download_company_reports(company_code, start_date, end_date, progress_callback=None):
    """
    Function to download reports for a specific company within a date range.
    
    Args:
        company_code: Company symbol (e.g., "REXP.N0000" or "DIPD.N0000")
        start_date: Start date as datetime.date object
        end_date: End date as datetime.date object
        progress_callback: Optional callback function to report progress
        
    Returns:
        Dictionary with results of the download operation
    """
    # Convert date range to years
    years = list(range(start_date.year, end_date.year + 1))
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback(0.2)
    
    # Download reports
    result = download_reports(company_code, years)
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback(1.0)
    
    return result

# For testing in standalone mode
if __name__ == "__main__":
    # Example usage - can be replaced with any direct function call
    # This is just for testing the script directly without Streamlit
    company_symbols = ["REXP.N0000", "DIPD.N0000"]
    # Use default years (last 3)
    result = download_reports(company_symbols)
    
    # Or specify years
    # result = download_reports(company_symbols, [2023, 2022, 2021])
