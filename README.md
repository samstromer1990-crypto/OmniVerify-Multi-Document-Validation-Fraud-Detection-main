# OmniVerify – AI-Based Document Verification & Fraud Risk Analysis

**OmniVerify** is a multi-document analysis system designed to validate document structure, extract information, inspect QR codes, analyze image quality, and estimate potential document manipulation risk.

The application supports **PAN Cards, Aadhaar Cards, and Driving Licences**. It combines computer vision, OCR, rule-based validation, and machine learning to generate an explainable document assessment.

> **Important:** OmniVerify performs document-level validation and risk analysis. It does not verify the identity of the document holder or establish official government authenticity.

---

## Key Features

### 1. Multi-Document Support
- Supports PAN Cards, Aadhaar Cards, and Driving Licences.
- Includes an automatic document-type selection option.
- Accepts document images in JPG, JPEG, and PNG formats.

### 2. OCR-Based Text Extraction
- Extracts text from uploaded document images using Tesseract OCR.
- Provides extracted text and OCR diagnostics.
- Displays word count and average OCR confidence.

### 3. QR Code Detection
- Detects QR codes present in document images.
- Uses QR information as part of the document validation workflow.

### 4. Structural Document Validation
- Examines document structure and required information.
- Combines OCR results, QR detection, and image-quality information.
- Generates a validation report.

### 5. Machine Learning-Based Fraud Risk Analysis
- Uses a trained machine learning model to estimate document manipulation risk.
- Classifies results into LOW and HIGH risk categories.
- Displays class probabilities and relevant extracted features.

### 6. Explainable AI with SHAP
- Integrates SHAP to provide feature-level explanations for model predictions.
- Shows which extracted features contribute towards genuine or elevated-risk classifications.

### 7. Image Quality Analysis
- Analyzes uploaded document images.
- Provides image-quality diagnostics as part of the verification workflow.

### 8. Verification Report
- Presents the document validation result and fraud-risk assessment.
- Includes technical diagnostics, OCR information, image-quality analysis, and a raw validation report.

---

## System Workflow

```text
       Document Image Upload
                 |
                 v
       Image Preprocessing
                 |
          +------+------+
          |             |
          v             v
      OCR Extraction  QR Detection
          |             |
          +------+------+
                 |
                 v
       Structural Validation
                 |
                 v
       Feature Extraction
                 |
                 v
       ML Fraud Risk Analysis
                 |
                 v
          SHAP Explanation
                 |
                 v
       Verification Assessment
                 |
                 v
       Risk Classification
          LOW / HIGH
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Image Processing | OpenCV |
| OCR | Tesseract OCR, Pytesseract |
| Numerical Processing | NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Containerization | Docker |

---

## Project Structure

```text
OmniVerify/
│
├── .streamlit/
│
├── data/
│   ├── samples/
│   └── synthetic_docs/
│
├── models/
│   └── fraud_model.pkl
│
├── reports/
│
├── src/
│
├── tests/
│
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.11
- Tesseract OCR
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/samstromer1990-crypto/OmniVerify-Multi-Document-Validation-Fraud-Detection-main.git
```

### 2. Navigate to the Project Directory

```bash
cd OmniVerify-Multi-Document-Validation-Fraud-Detection-main
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Tesseract OCR

Tesseract OCR must be installed separately on your system.

- **Windows:** Install Tesseract OCR and ensure its executable is accessible to the application.
- **Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Running with Docker

The repository includes a Dockerfile for containerized execution.

Build the Docker image:

```bash
docker build -t omniverify .
```

Run the container:

```bash
docker run -p 8501:8501 omniverify
```

Open the application at:

```text
http://localhost:8501
```

---

## Model and Data

OmniVerify uses a trained fraud-risk classification model stored in the `models/` directory.

The project also includes synthetic document data organized into genuine and manipulated categories, which is used by the application's SHAP explanation workflow.

The application expects the required model and supporting project data to be available in their respective directories.

---

## Limitations

- The system analyzes document images; it does not connect to official government databases.
- A LOW-risk result does not guarantee that a document is authentic.
- OCR accuracy depends on image quality, resolution, orientation, and text clarity.
- Fraud-risk predictions are model-based assessments and should not be treated as definitive proof of forgery.
- The system is intended as a decision-support and technical demonstration project, not a replacement for official document verification.

---

## Project Objective

The objective of OmniVerify is to explore how computer vision, OCR, machine learning, and explainable AI can work together in a document-analysis pipeline.

The project focuses on combining multiple forms of document inspection into one application and presenting the resulting assessment in an understandable format.

---

## Author

**Shaghil Ansari**

Computer Science and Engineering Student

[GitHub](https://github.com/samstromer1990-crypto)

---

*Developed as an AI and Computer Vision project exploring intelligent document analysis and fraud-risk assessment.*
