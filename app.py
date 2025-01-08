import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import openai
import mimetypes
import pandas as pd
import time
from io import StringIO
from PyPDF2 import PdfReader
import requests

print("Starting application...")

def cleanup_old_files():
    """Clean up any files that couldn't be deleted in previous sessions"""
    cleanup_file = 'cleanup.txt'
    if os.path.exists(cleanup_file):
        try:
            with open(cleanup_file, 'r') as f:
                files_to_cleanup = f.readlines()
            
            # Remove the cleanup file first
            os.remove(cleanup_file)
            
            # Try to remove each file
            for file_path in files_to_cleanup:
                file_path = file_path.strip()
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up old file: {file_path}")
                    except Exception as e:
                        print(f"Failed to clean up {file_path}: {str(e)}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

# Run cleanup on startup
cleanup_old_files()

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# API Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client
try:
    openai.api_key = API_KEY
    print("OpenAI API key set successfully")
except Exception as e:
    print(f"Error setting OpenAI API key: {str(e)}")
    raise

# Function to get organization ID from API key
def get_organization_id():
    try:
        # Try to make a simple request to get organization info
        models = openai.Model.list()
        # If successful, get the organization ID from the response headers
        if hasattr(models, 'headers') and 'x-organization-id' in models.headers:
            return models.headers['x-organization-id']
    except Exception as e:
        error_message = str(e)
        if 'mismatched_organization' in error_message:
            # Extract organization ID from error message if possible
            import re
            match = re.search(r'org-[a-zA-Z0-9]+', error_message)
            if match:
                return match.group(0)
    return None

# Get the correct organization ID
org_id = get_organization_id()
if org_id:
    print(f"Found organization ID: {org_id}")
    openai.organization = org_id
else:
    print("Warning: Could not determine organization ID")

# Validate API key on startup
try:
    models = openai.Model.list()
    print("\nOpenAI API Configuration:")
    print(f"API Key Type: Project-scoped")
    print(f"Organization ID: {org_id}")
    print("\nAPI connection successful")
    print("Available models:", [model.id for model in models.data])
except Exception as e:
    print("\nAPI Key Validation Error:")
    print("Error type:", type(e).__name__)
    print("Error message:", str(e))
    if hasattr(e, 'response'):
        print("Status code:", getattr(e.response, 'status_code', None))
        print("Headers:", getattr(e.response, 'headers', None))
        try:
            print("Content:", getattr(e.response, 'content', None))
        except:
            print("Content: Unable to read response content")

def truncate_content(content, max_chars=4000):
    """Truncate content to avoid token limits"""
    if len(content) > max_chars:
        return content[:max_chars] + "\n[Content truncated due to length...]"
    return content

def call_openai_with_retry(messages, max_retries=3):
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,  # Limit response tokens
                temperature=0.7
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"OpenAI API Error (Attempt {attempt + 1}/{max_retries}): {str(e)}")
            error_message = str(e)
            
            # Check for specific error types
            if "insufficient_quota" in error_message or "exceeded your current quota" in error_message:
                print("Usage limit error detected")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
            elif "invalid_api_key" in error_message or "mismatched_organization" in error_message:
                print("Authentication error detected")
                raise e
            elif "model_not_found" in error_message:
                # Try with a different model if gpt-4 is not available
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",  # Fallback model
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return response.choices[0].message['content']
                except:
                    pass  # If fallback fails, continue with the retry loop
            
            # If we've exhausted all retries, raise the last error
            if attempt == max_retries - 1:
                raise e
            
            # Wait before next retry
            time.sleep(1)

def is_valid_file(filename):
    # Check if file extension is .csv, .txt, .pdf, or .xlsx
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'txt', 'pdf', 'xlsx'}

def read_pdf_file(filepath):
    try:
        # Create a PDF reader object
        reader = PdfReader(filepath)
        
        # Extract text from all pages
        text = []
        text.append("PDF File Analysis:")
        text.append(f"\nTotal Pages: {len(reader.pages)}")
        
        # Extract text from each page
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                text.append(f"\n--- Page {i+1} ---")
                text.append(page_text.strip())
        
        # Join all text parts
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return f"Error processing PDF: {str(e)}"

def read_excel_file(filepath):
    try:
        # Read Excel file with pandas
        xl = None
        df = None
        summary = []
        
        try:
            # Read the Excel file
            xl = pd.ExcelFile(filepath)
            sheet_names = xl.sheet_names
            
            # Format the data
            summary.append("Excel File Analysis:")
            summary.append(f"\nSheets in workbook: {', '.join(sheet_names)}")
            
            # Read the first sheet
            df = xl.parse(sheet_names[0])
            
            # Analyze active sheet
            summary.append(f"\nActive Sheet Analysis:")
            summary.append(f"\nColumn Names: {', '.join(df.columns.tolist())}")
            
            summary.append("\nData Preview (First 5 rows):")
            summary.append(df.head().to_string())
            
            summary.append("\nBasic Statistics:")
            summary.append(df.describe().to_string())
            
            summary.append("\nData Info:")
            buffer = StringIO()
            df.info(buf=buffer)
            summary.append(buffer.getvalue())
            buffer.close()
            
            return "\n".join(summary)
            
        finally:
            # Ensure we close all file handles
            if xl is not None:
                xl.close()
            # Force garbage collection of DataFrame to release file handle
            if df is not None:
                del df
            
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return f"Error processing Excel file: {str(e)}"

def read_and_format_file(filepath):
    file_ext = filepath.rsplit('.', 1)[1].lower()
    
    if file_ext == 'pdf':
        return read_pdf_file(filepath)
    elif file_ext == 'xlsx':
        return read_excel_file(filepath)
    elif file_ext == 'csv':
        try:
            # Try reading with pandas default engine
            df = pd.read_csv(filepath, encoding_errors='replace')
        except Exception as e:
            try:
                # Try reading with python engine which is more forgiving
                df = pd.read_csv(filepath, encoding_errors='replace', engine='python')
            except Exception as e:
                print(f"Error reading CSV with python engine: {str(e)}")
                return read_text_file(filepath)

        try:
            # Format the data
            summary = []
            summary.append("CSV File Analysis:")
            summary.append(f"\nColumn Names: {', '.join(df.columns.tolist())}")
            
            summary.append("\nData Preview (First 5 rows):")
            summary.append(df.head().to_string())
            
            summary.append("\nBasic Statistics:")
            summary.append(df.describe().to_string())
            
            summary.append("\nData Info:")
            buffer = StringIO()
            df.info(buf=buffer)
            summary.append(buffer.getvalue())
            buffer.close()
            
            return "\n".join(summary)
        except Exception as e:
            print(f"Error formatting CSV data: {str(e)}")
            # If formatting fails, return raw text
            return df.to_string()
    else:
        return read_text_file(filepath)

def read_text_file(filepath):
    try:
        # Try reading with default encoding
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file: {str(e)}")
        # If all else fails, try binary read and decode
        with open(filepath, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')

def check_api_quota():
    """Check if the API key has available quota"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        error_message = str(e)
        if "insufficient_quota" in error_message or "exceeded your current quota" in error_message:
            return False
        raise e

def get_data_analysis_prompt(file_content):
    """Prompt for analyzing tabular data (CSV, XLSX)"""
    return f"""You are a senior data analyst. Please provide a comprehensive analysis of the dataset in the following format:

1. Executive Summary (2-3 paragraphs):
   - Provide a high-level overview of what the dataset contains
   - Highlight the most interesting and significant findings
   - Discuss key patterns, trends, and relationships discovered
   - Emphasize practical implications and insights
   - Note any significant limitations or considerations

2. Statistical Overview:
   A. Dataset Dimensions
   - Total number of records
   - Number of variables
   - Data completeness metrics
   
   B. Numerical Variables
   - List each numerical variable with:
     * Mean, median, standard deviation
     * Range (min, max)
     * Distribution characteristics
     * Notable outliers
   
   C. Categorical Variables
   - List each categorical variable with:
     * Frequency distribution
     * Mode and unique values count
     * Missing value percentage

3. Key Relationships:
   - List the strongest correlations found
   - Describe any notable patterns
   - Highlight unexpected findings

4. Recommended Visualizations:
   For each suggested visualization, provide:
   - Type of plot (e.g., scatter plot, bar chart, etc.)
   - Variables to include
   - Purpose of the visualization
   - Key insights it would reveal
   Example format:
   1. [Plot Type]: [Variables] - [Purpose]
   2. [Plot Type]: [Variables] - [Purpose]

5. Action Items:
   - List 3-5 concrete recommendations
   - Include specific metrics to track
   - Suggest next steps for deeper analysis

Here's the data to analyze:

{file_content}"""

def get_text_analysis_prompt(file_content):
    """Prompt for analyzing text documents (TXT, PDF)"""
    return f"""You are a senior content analyst. Please provide a clear and practical summary of this document in the following format:

1. Executive Summary (2-3 paragraphs):
   - Provide a clear, concise overview of what this document is about
   - Highlight the main purpose and key points
   - Identify who this document is for and why it matters

2. Key Information:
   - Main topics covered
   - Important dates or deadlines (if any)
   - Key people, organizations, or entities mentioned
   - Critical numbers or statistics (if any)
   - Important definitions or terms explained

3. Instructions or Action Items (if present):
   [Include this section only if the document contains instructions or required actions]
   - List all instructions or required actions clearly
   - Specify any step-by-step processes
   - Note any deadlines or time-sensitive items
   - Highlight any requirements or prerequisites
   - List any warnings or important cautions

4. Important Details:
   - List any significant findings or conclusions
   - Note any crucial requirements or conditions
   - Highlight any exceptions or special cases
   - Include any relevant contact information or resources

5. Additional Notes:
   - Any supplementary information that might be useful
   - References to other documents or resources
   - Potential next steps or follow-up actions
   - Questions that might need clarification

Please format your response using clear headings and maintain readability with appropriate spacing. Use bullet points for lists and highlight particularly important information. If certain sections are not applicable to this document, you may skip them.

Here's the text to analyze:

{file_content}"""

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def call_ollama(prompt):
    """Call the local Ollama instance"""
    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json()['response']
    except requests.RequestException as e:
        print(f"Error calling Ollama: {str(e)}")
        raise Exception("Error connecting to local LLM. Please ensure Ollama is running.") from e

def analyze_with_llm(prompt, safe_mode=False):
    """Analyze text using either OpenAI or local Llama"""
    if safe_mode:
        print("Using local Llama for analysis...")
        try:
            return call_ollama(prompt)
        except Exception as e:
            print(f"Error with local LLM: {str(e)}")
            raise
    else:
        print("Using OpenAI for analysis...")
        try:
            response = call_openai_with_retry([
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": prompt}
            ])
            return response
        except Exception as e:
            print(f"Error with OpenAI: {str(e)}")
            raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    filepath = None
    try:
        # Check if safe mode is enabled
        safe_mode = request.form.get('safe_mode') == 'true'
        print(f"Safe mode enabled: {safe_mode}")

        if not safe_mode and not check_api_quota():
            return jsonify({
                'error': 'OpenAI API quota exceeded. Please check your billing status or enable Safe Mode to use local LLM.'
            }), 429

        print("Received request files:", request.files)
        if 'file' not in request.files:
            print("No file found in request.files")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print("Filename:", file.filename)
        if file.filename == '':
            print("Empty filename detected")
            return jsonify({'error': 'No file selected'}), 400

        if not is_valid_file(file.filename):
            return jsonify({'error': 'Please upload a CSV, Excel, text, or PDF file'}), 400

        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Attempting to save file to: {filepath}")
        file.save(filepath)

        try:
            file_content = read_and_format_file(filepath)
            # Truncate content if too long
            file_content = truncate_content(file_content)
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 400

        # Choose appropriate prompt based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in ['csv', 'xlsx']:
            prompt = get_data_analysis_prompt(file_content)
        else:  # txt or pdf
            prompt = get_text_analysis_prompt(file_content)

        try:
            # Use either OpenAI or local Llama based on safe_mode
            analysis = analyze_with_llm(prompt, safe_mode)
            return jsonify({'analysis': analysis})
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message.lower() or "exceeded your current quota" in error_message:
                return jsonify({
                    'error': 'OpenAI API quota exceeded. Please enable Safe Mode to use local LLM.'
                }), 429
            return jsonify({'error': f'Analysis Error: {error_message}'}), 500

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up the temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")
                with open('cleanup.txt', 'a') as f:
                    f.write(f"{filepath}\n")

if __name__ == '__main__':
    app.run(debug=True)
