import os
from pathlib import Path
from dotenv import load_dotenv
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Constants for folder structure
COMPANY_FOLDERS = {
    "REXP.N0000": "Company1_Reports",
    "DIPD.N0000": "Company2_Reports"
}

# Create dataset folders for each company
DATASET_FOLDERS = {
    "REXP.N0000": "Company1_Datasets",
    "DIPD.N0000": "Company2_Datasets"
}

# Create dataset folders if they don't exist
for folder in DATASET_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

def process_company_reports(company_folder, dataset_folder):
    """Process all PDF reports for a company"""
    print(f"\nProcessing reports from: {company_folder}")
    
    # Set up parser
    parser = LlamaParse(
        result_type="text",  # Using text instead of markdown for simpler output
        api_key=os.environ.get("LLAMA_CLOUD_API_KEY")
    )
    
    # Configure file extractor for PDFs
    file_extractor = {".pdf": parser}
    
    try:
        # Use SimpleDirectoryReader to parse all PDFs in the folder
        documents = SimpleDirectoryReader(
            input_dir=company_folder,
            file_extractor=file_extractor
        ).load_data()
        
        # Save each document
        for i, doc in enumerate(documents):
            output_file = os.path.join(dataset_folder, f"parsed_doc_{i+1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(doc.text)
            print(f"Saved parsed document to: {output_file}")
            
    except Exception as e:
        print(f"Error processing reports: {str(e)}")

def main():
    """Main function to process all company reports"""
    for company_code, folder in COMPANY_FOLDERS.items():
        if os.path.exists(folder):
            dataset_folder = DATASET_FOLDERS[company_code]
            process_company_reports(folder, dataset_folder)
        else:
            print(f"Folder not found: {folder}")

if __name__ == "__main__":
    main()