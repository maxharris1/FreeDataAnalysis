import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI, OpenAIError
import mimetypes
import pandas as pd
import time
from io import StringIO

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# API Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("OpenAI API key not found in .env file. Please add OPENAI_API_KEY=your-api-key to .env file")

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.openai.com/v1"
)

# Function to get organization ID from API key
def get_organization_id():
    try:
        # Try to make a simple request to get organization info
        models = client.models.list()
        # If successful, get the organization ID from the response headers
        response_headers = getattr(models, 'response', None)
        if response_headers and 'x-organization-id' in response_headers.headers:
            return response_headers.headers['x-organization-id']
    except OpenAIError as e:
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
    # Reinitialize client with the correct organization ID
    client = OpenAI(
        api_key=API_KEY,
        organization=org_id,
        base_url="https://api.openai.com/v1"
    )
else:
    print("Warning: Could not determine organization ID")

# Validate API key on startup
try:
    response = client.models.list()
    print("\nOpenAI API Configuration:")
    print(f"API Key Type: Project-scoped")
    print(f"Organization ID: {org_id}")
    print("\nAPI connection successful")
    print("Available models:", [model.id for model in response.data])
except OpenAIError as e:
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
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,  # Limit response tokens
                temperature=0.7
            )
            return response.choices[0].message.content
        except OpenAIError as e:
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
                # Try with a different model if gpt-3.5-turbo is not available
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-16k",  # Fallback model
                        messages=messages,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                except:
                    pass  # If fallback fails, continue with the retry loop
            
            # If we've exhausted all retries, raise the last error
            if attempt == max_retries - 1:
                raise e
            
            # Wait before next retry
            time.sleep(1)

def is_valid_file(filename):
    # Check if file extension is .csv or .txt
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'txt'}

def read_and_format_file(filepath):
    file_ext = filepath.rsplit('.', 1)[1].lower()
    
    if file_ext == 'csv':
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except OpenAIError as e:
        error_message = str(e)
        if "insufficient_quota" in error_message or "exceeded your current quota" in error_message:
            return False
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # First check API quota
    if not check_api_quota():
        return jsonify({
            'error': 'OpenAI API quota exceeded. Please check your billing status at https://platform.openai.com/account/billing'
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
        return jsonify({'error': 'Please upload a CSV or text file'}), 400

    try:
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

        # Create analysis prompt
        prompt = f"""You are a senior data analyst. Please provide a comprehensive analysis of the dataset in the following format:

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

Please format your response using clear headings and maintain readability with appropriate spacing. Use bullet points for lists and include specific numbers and percentages where relevant.

Here's the data to analyze:

{file_content}"""

        try:
            # Call OpenAI API with retry logic
            analysis = call_openai_with_retry([
                {"role": "system", "content": "You are a helpful data analysis assistant."},
                {"role": "user", "content": prompt}
            ])
        except OpenAIError as e:
            error_message = str(e)
            if "insufficient_quota" in error_message.lower() or "exceeded your current quota" in error_message:
                return jsonify({
                    'error': 'OpenAI API quota exceeded. Please check your billing status at https://platform.openai.com/account/billing'
                }), 429
            return jsonify({'error': f'OpenAI API Error: {error_message}'}), 500

        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({'analysis': analysis})

    except Exception as e:
        # Add detailed error logging
        print(f"Error during analysis: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
