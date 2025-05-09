from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for Word processing
import pandas as pd  # For CSV and Excel processing
import openai
from werkzeug.utils import secure_filename
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
CORS(app)

# Azure OpenAI Configuration from .env
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

UPLOAD_FOLDER = "upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

uploaded_filename = ""  # Store uploaded file name
document_text = ""  # Store extracted text


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload and extracts text."""
    global uploaded_filename, document_text

    if "file" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        uploaded_filename = filename
        document_text = extract_text(file_path)

        if document_text.strip():
            return jsonify({"message": "File uploaded successfully", "redirect": "process.html"}), 200
        else:
            return jsonify({"message": "File uploaded but no readable content found"}), 400

    except Exception as e:
        return jsonify({"message": "File upload failed", "error": str(e)}), 500


def extract_text(file_path):
    """Extracts text from PDF, DOCX, CSV, or Excel."""
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".csv"):
        return extract_text_from_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return extract_text_from_excel(file_path)
    return ""


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n"
    return text


def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_csv(file_path):
    """Extract text from a CSV file."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df.to_string(index=False)  # Convert CSV to text
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def extract_text_from_excel(file_path):
    """Extract text from an Excel file."""
    try:
        df = pd.read_excel(file_path)  # Read Excel file
        return df.to_string(index=False)  # Convert Excel data to text
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"


@app.route("/process", methods=["POST"])
def process_file():
    """Processes the uploaded document and returns its analysis result."""
    global document_text

    if not document_text:
        return jsonify({"message": "No document analyzed yet"}), 400

    word_count = len(document_text.split())
    analysis_result = f"Analysis successful! Word count: {word_count}"

    return jsonify({"message": analysis_result, "content": document_text[:500]}), 200  # Show first 500 chars


@app.route("/ask", methods=["POST"])
def ask_question():
    """Answers user questions based on extracted document content."""
    global document_text

    if not document_text:
        return jsonify({"message": "No document analyzed yet"}), 400

    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"message": "Please ask a question"}), 400

    # Query GPT-4 with document context
    answer = query_gpt4(question, document_text)

    return jsonify({"question": question, "answer": answer})


def query_gpt4(question, context):
    """Query GPT-4 model using Azure OpenAI."""
    try:
        client = openai.AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION
        )

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are Financial assistant that answers questions based on the uploaded document."},
                {"role": "user", "content": f"Document: {context[:2000]}...\n\nQuestion: {question}"}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying GPT-4: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True, port=5000)
