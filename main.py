from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json
import os
import tempfile
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from jinja2 import Template
import io
import base64
from dotenv import load_dotenv
from reportlab.platypus import Image
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from typing import List
from dotenv import load_dotenv

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Data Preparation API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store data between requests
current_data = None
current_template = None
processing_steps = []

# Templates for different data types
TEMPLATES = {
    "household": [
        "household_id", "member_id", "age", "gender", "education", "occupation",
        "income", "expenditure", "food_expenditure", "housing_expenditure",
        "transport_expenditure", "health_expenditure", "education_expenditure",
        "entertainment_expenditure", "clothing_expenditure", "utilities_expenditure",
        "household_size", "children_count", "elderly_count", "working_members",
        "rural_urban", "state", "district", "village", "caste", "religion",
        "land_ownership", "house_ownership", "vehicle_ownership", "bank_account",
        "insurance_coverage", "pension_coverage", "health_insurance", "loan_amount",
        "savings_amount", "investment_amount", "debt_amount", "asset_value",
        "monthly_income", "annual_income", "poverty_line", "consumption_quintile"
    ],
    "industrial": [
        "company_id", "industry_type", "employee_count", "revenue", "profit",
        "assets", "liabilities", "equity", "market_cap", "stock_price"
    ],
    "employment": [
        "employee_id", "job_title", "department", "salary", "experience_years",
        "education_level", "skills", "performance_rating", "hire_date"
    ],
    "other": []
}

# Validation rules
VALIDATION_RULES = {
    "general": {
        "age": ["age > 0", "age < 120"],
        "income": ["income >= 0"],
        "expenditure": ["expenditure >= 0"],
        "household_size": ["household_size > 0", "household_size < 20"]
    },
    "statistical": {
        "iqr_multiplier": 1.5,
        "z_score_threshold": 3.0
    }
}

@app.get("/")
async def root():
    return {"message": "Data Preparation API is running"}

@app.get("/sample-data")
async def get_sample_data():
    """Download sample data for testing"""
    try:
        sample_file_path = "sample_data.csv"
        if os.path.exists(sample_file_path):
            return FileResponse(
                sample_file_path,
                media_type='text/csv',
                filename='sample_data.csv'
            )
        else:
            raise HTTPException(status_code=404, detail="Sample data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving sample data: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_data, processing_steps
    
    if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and convert to CSV if needed
        if file.filename.endswith('.json'):
            # Convert JSON to CSV
            json_data = json.loads(content.decode('utf-8'))
            if isinstance(json_data, list):
                df = pd.DataFrame(json_data)
            else:
                df = pd.DataFrame([json_data])
            
            # Save as CSV in temp file
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_csv.name, index=False)
            temp_csv.close()
            
            current_data = df
            processing_steps.append(f"Uploaded JSON file '{file.filename}' and converted to CSV")
            
            return {
                "message": "JSON file uploaded and converted to CSV successfully",
                "columns": df.columns.tolist(),
                "rows": len(df),
                "temp_file": temp_csv.name
            }
        
        elif file.filename.endswith('.csv'):
            # Read CSV directly
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            current_data = df
            processing_steps.append(f"Uploaded CSV file '{file.filename}'")
            
            return {
                "message": "CSV file uploaded successfully",
                "columns": df.columns.tolist(),
                "rows": len(df)
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or JSON.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/templates")
async def get_templates():
    return {"templates": list(TEMPLATES.keys())}

@app.get("/template/{template_name}")
async def get_template_columns(template_name: str):
    if template_name not in TEMPLATES:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"columns": TEMPLATES[template_name]}

@app.post("/map-schema")
async def map_schema(template_name: str = Form(...)):
    global current_data, current_template
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a file first.")
    
    if template_name not in TEMPLATES:
        raise HTTPException(status_code=400, detail="Invalid template name")
    
    current_template = template_name
    template_columns = TEMPLATES[template_name]
    data_columns = current_data.columns.tolist()
    
    # Perform schema mapping using regex and cosine similarity
    mappings = []
    
    for template_col in template_columns:
        best_match = None
        best_score = 0
        
        for data_col in data_columns:
            # Simple regex matching
            if template_col.lower() in data_col.lower() or data_col.lower() in template_col.lower():
                score = 0.8
            else:
                # Calculate cosine similarity (simplified)
                score = calculate_similarity(template_col, data_col)
            
            if score > best_score:
                best_score = score
                best_match = data_col
        
        mappings.append({
            "template_column": template_col,
            "mapped_column": best_match,
            "confidence_score": round(best_score, 3),
            "available_columns": data_columns
        })
    
    processing_steps.append(f"Schema mapped using '{template_name}' template")
    
    return {
        "mappings": mappings,
        "template_name": template_name
    }

@app.post("/validate")
async def validate_data(rules: str = Form(...)):
    global current_data, processing_steps
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        # Parse custom rules
        custom_rules = json.loads(rules) if rules else {}
        
        # Apply validation rules
        validation_results = apply_validation_rules(current_data, custom_rules)
        
        # Remove invalid data
        initial_rows = len(current_data)
        current_data = remove_invalid_data(current_data, validation_results)
        final_rows = len(current_data)
        
        processing_steps.append(f"Data validation completed. Removed {initial_rows - final_rows} invalid rows")
        
        return {
            "validation_results": validation_results,
            "rows_removed": initial_rows - final_rows,
            "remaining_rows": final_rows
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

@app.post("/eda")
async def generate_eda():
    global current_data, processing_steps
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        # Check if data is valid
        if current_data.empty:
            raise HTTPException(status_code=400, detail="Uploaded data is empty")
        
        # Generate custom EDA report
        eda_results = generate_custom_eda(current_data)
        
        # Check if EDA generation failed
        if "error" in eda_results:
            raise HTTPException(status_code=500, detail=f"EDA analysis failed: {eda_results['error']}")
        
        # Save HTML report
        report_path = "temp_eda_report.html"
        save_eda_html_report(eda_results, report_path)
        
        processing_steps.append("Exploratory Data Analysis (EDA) report generated")
        
        return {
            "message": "EDA report generated successfully",
            "report_path": report_path,
            "data_shape": current_data.shape,
            "columns": current_data.columns.tolist(),
            "eda_summary": eda_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in EDA generation: {e}")
        raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")

@app.post("/impute")
async def impute_missing_values(columns: str = Form(...), method: str = Form(...)):
    global current_data, processing_steps
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        columns_list = json.loads(columns)
        
        # Apply imputation
        current_data = apply_imputation(current_data, columns_list, method)
        
        processing_steps.append(f"Missing value imputation applied using {method} method")
        
        return {
            "message": f"Imputation completed using {method} method",
            "data_shape": current_data.shape
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Imputation error: {str(e)}")

@app.post("/detect-outliers")
async def detect_outliers(columns: str = Form(...), method: str = Form(...)):
    global current_data, processing_steps
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        columns_list = json.loads(columns)
        
        # Apply outlier detection
        outlier_results = apply_outlier_detection(current_data, columns_list, method)
        
        processing_steps.append(f"Outlier detection completed using {method} method")
        
        return {
            "outlier_results": outlier_results,
            "method_used": method
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outlier detection error: {str(e)}")

@app.post("/generate-report")
async def generate_report():
    global current_data, processing_steps
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        # Generate LLM summary using Gemini API
        summary = generate_llm_summary(current_data, processing_steps)
        
        # Generate PDF report
        pdf_path = generate_pdf_report(summary, current_data, processing_steps)
        
        processing_steps.append("LLM-based report generated and converted to PDF")
        
        return {
            "message": "Report generated successfully",
            "pdf_path": pdf_path,
            "summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

@app.get("/eda-report")
async def get_eda_report():
    """Get the generated EDA HTML report"""
    try:
        report_path = "temp_eda_report.html"
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            raise HTTPException(status_code=404, detail="EDA report not found. Please generate EDA first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving EDA report: {str(e)}")

@app.get("/download/{file_type}")
async def download_file(file_type: str):
    global current_data
    
    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available for download")
    
    try:
        if file_type == "csv":
            # Save as CSV
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            current_data.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type='text/csv',
                filename='processed_data.csv'
            )
        
        elif file_type == "excel":
            # Save as Excel
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False)
            current_data.to_excel(temp_file.name, index=False)
            temp_file.close()
            
            return FileResponse(
                temp_file.name,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                filename='processed_data.xlsx'
            )
        
        elif file_type == "pdf":
            # Generate and download PDF report
            try:
                summary = generate_llm_summary(current_data, processing_steps)
                pdf_path = generate_pdf_report(summary, current_data, processing_steps)
                
                if os.path.exists(pdf_path):
                    return FileResponse(
                        pdf_path,
                        media_type='application/pdf',
                        filename='data_preprocessing_report.pdf'
                    )
                else:
                    raise HTTPException(status_code=500, detail="PDF generation failed")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

# Helper functions
def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    # Exact match
    if str1_lower == str2_lower:
        return 1.0
    
    # Contains match
    if str1_lower in str2_lower or str2_lower in str1_lower:
        return 0.7
    
    # Word overlap
    words1 = set(str1_lower.split())
    words2 = set(str2_lower.split())
    
    if words1 and words2:
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        return overlap / total if total > 0 else 0.0
    
    return 0.0

def apply_validation_rules(df: pd.DataFrame, custom_rules: Dict) -> Dict:
    """Apply validation rules to the dataset"""
    results = {}

    for column in df.columns:
        column_results = []

        # Apply general rules
        if column in VALIDATION_RULES["general"]:
            # Ensure numeric conversion for numeric checks
            df[column] = pd.to_numeric(df[column], errors='coerce')

            for rule in VALIDATION_RULES["general"][column]:
                if ">=" in rule:
                    threshold = float(rule.split(">=")[1].strip())
                    invalid_count = len(df[df[column] < threshold])
                elif "<=" in rule:
                    threshold = float(rule.split("<=")[1].strip())
                    invalid_count = len(df[df[column] > threshold])
                elif ">" in rule:
                    threshold = float(rule.split(">")[1].strip())
                    invalid_count = len(df[df[column] <= threshold])
                elif "<" in rule:
                    threshold = float(rule.split("<")[1].strip())
                    invalid_count = len(df[df[column] >= threshold])
                else:
                    invalid_count = 0  # fallback

                column_results.append({
                    "rule": rule,
                    "invalid_count": invalid_count,
                    "passed": invalid_count == 0
                })

        # Apply custom rules
        if column in custom_rules:
            for rule in custom_rules[column]:
                column_results.append({
                    "rule": rule,
                    "invalid_count": 0,
                    "passed": True
                })

        results[column] = column_results

    return results

def remove_invalid_data(df: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
    """Remove rows that fail validation rules"""
    mask = pd.Series(True, index=df.index)

    for column, rules in validation_results.items():
        if column in df.columns:
            for rule in rules:
                if not rule["passed"]:
                    expr = rule["rule"]
                    if ">=" in expr:
                        threshold = float(expr.split(">=")[1].strip())
                        mask &= (df[column] >= threshold)
                    elif "<=" in expr:
                        threshold = float(expr.split("<=")[1].strip())
                        mask &= (df[column] <= threshold)
                    elif ">" in expr:
                        threshold = float(expr.split(">")[1].strip())
                        mask &= (df[column] > threshold)
                    elif "<" in expr:
                        threshold = float(expr.split("<")[1].strip())
                        mask &= (df[column] < threshold)

    return df[mask]

def apply_imputation(df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
    """Apply missing value imputation"""
    df_copy = df.copy()
    
    for column in columns:
        if column in df_copy.columns and df_copy[column].isnull().any():
            if method == "mean":
                imputer = SimpleImputer(strategy='mean')
            elif method == "median":
                imputer = SimpleImputer(strategy='median')
            elif method == "knn":
                imputer = KNNImputer(n_neighbors=5)
            else:
                continue
            
            df_copy[column] = imputer.fit_transform(df_copy[[column]])
    
    return df_copy

def apply_outlier_detection(df: pd.DataFrame, columns: List[str], method: str) -> Dict:
    """Apply outlier detection methods"""
    results = {}
    
    for column in columns:
        if column in df.columns and df[column].dtype in ['int64', 'float64']:
            if method == "iqr":
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
                results[column] = {
                    "method": "IQR",
                    "outlier_count": len(outliers),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                }
            
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outliers = df[z_scores > 3]
                
                results[column] = {
                    "method": "Z-Score",
                    "outlier_count": len(outliers),
                    "threshold": 3
                }
            
            elif method == "isolation_forest":
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(df[[column]])
                outlier_indices = np.where(outliers == -1)[0]
                
                results[column] = {
                    "method": "Isolation Forest",
                    "outlier_count": len(outlier_indices),
                    "contamination": 0.1
                }
    
    return results

def generate_llm_summary(df: pd.DataFrame, steps: List[str]) -> str:
    """Generate LLM summary using Gemini API"""
    try:
        # Set up Gemini API (you'll need to set GOOGLE_API_KEY environment variable)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "LLM summary not available - API key not configured"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        # Create prompt
        prompt = f"""
        Generate a comprehensive summary of the data preprocessing steps taken:
        
        Dataset Information:
        - Shape: {df.shape}
        - Columns: {', '.join(df.columns.tolist())}
        
        Processing Steps:
        {chr(10).join(f"- {step}" for step in steps)}
        
        Please provide a professional summary including:
        1. Overview of the dataset
        2. Summary of preprocessing steps
        3. Key insights and recommendations
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"LLM summary generation failed: {str(e)}"

def generate_custom_eda(df: pd.DataFrame) -> Dict:
    """Generate custom EDA analysis without external dependencies"""
    try:
        # Basic safety checks
        if df is None or df.empty:
            return {"error": "DataFrame is empty or None"}
        
        eda_results = {
            "overview": {
                "shape": df.shape,
                "memory_usage": int(df.memory_usage(deep=True).sum()) if hasattr(df, 'memory_usage') else 0,
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            },
            "missing_values": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            "numeric_columns": {},
            "categorical_columns": {},
            "correlations": {}
        }
        
        # Analyze numeric columns safely
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    eda_results["numeric_columns"][str(col)] = {
                        "mean": float(col_data.mean()) if len(col_data) > 0 else None,
                        "median": float(col_data.median()) if len(col_data) > 0 else None,
                        "std": float(col_data.std()) if len(col_data) > 1 else None,
                        "min": float(col_data.min()) if len(col_data) > 0 else None,
                        "max": float(col_data.max()) if len(col_data) > 0 else None,
                        "unique_count": int(col_data.nunique()),
                        "missing_count": int(df[col].isnull().sum())
                    }
                else:
                    eda_results["numeric_columns"][str(col)] = {
                        "mean": None, "median": None, "std": None, "min": None, "max": None,
                        "unique_count": 0, "missing_count": int(df[col].isnull().sum())
                    }
        except Exception as e:
            print(f"Error analyzing numeric columns: {e}")
            eda_results["numeric_columns"] = {}
        
        # Analyze categorical columns safely
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    top_values = col_data.value_counts().head(5).to_dict()
                    eda_results["categorical_columns"][str(col)] = {
                        "unique_count": int(col_data.nunique()),
                        "missing_count": int(df[col].isnull().sum()),
                        "top_values": {str(k): int(v) for k, v in top_values.items()}
                    }
                else:
                    eda_results["categorical_columns"][str(col)] = {
                        "unique_count": 0, "missing_count": int(df[col].isnull().sum()), "top_values": {}
                    }
        except Exception as e:
            print(f"Error analyzing categorical columns: {e}")
            eda_results["categorical_columns"] = {}
        
        # Calculate correlations for numeric columns safely
        try:
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_results["correlations"] = {str(k): {str(k2): float(v2) for k2, v2 in v.items()} 
                                             for k, v in corr_matrix.to_dict().items()}
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            eda_results["correlations"] = {}
        
        return eda_results
        
    except Exception as e:
        print(f"Error in custom EDA: {e}")
        return {"error": str(e)}

def save_eda_html_report(eda_results: Dict, file_path: str):
    """Save EDA results as HTML report"""
    try:
        # Check if eda_results has an error
        if "error" in eda_results:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Profile Report - Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .error {{ background-color: #ffebee; padding: 20px; border-radius: 5px; color: #c62828; }}
                </style>
            </head>
            <body>
                <div class="error">
                    <h1>Error Generating Report</h1>
                    <p><strong>Error:</strong> {eda_results['error']}</p>
                    <p>Please check your data and try again.</p>
                </div>
            </body>
            </html>
            """
        else:
            # Safe access to eda_results with defaults
            overview = eda_results.get('overview', {})
            missing_values = eda_results.get('missing_values', {})
            numeric_columns = eda_results.get('numeric_columns', {})
            categorical_columns = eda_results.get('categorical_columns', {})
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Profile Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; }}
                    .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                    .table {{ border-collapse: collapse; width: 100%; }}
                    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .table th {{ background-color: #f2f2f2; }}
                    .numeric {{ background-color: #e8f5e8; }}
                    .categorical {{ background-color: #fff3cd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Profile Report</h1>
                    <p>Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="metric">
                        <p><strong>Shape:</strong> {overview.get('shape', [0, 0])[0]} rows × {overview.get('shape', [0, 0])[1]} columns</p>
                        <p><strong>Memory Usage:</strong> {overview.get('memory_usage', 0) / 1024:.2f} KB</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Missing Values</h2>
                    <div class="metric">
                        <p><strong>Total missing values:</strong> {sum(missing_values.values()) if missing_values else 0}</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Numeric Columns Analysis</h2>
                    {f'''
                    <table class="table numeric">
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Unique</th>
                            <th>Missing</th>
                        </tr>
                        {''.join([f'''
                        <tr>
                            <td>{col}</td>
                                                     <td>{f"{numeric_columns[col].get('mean', 'N/A'):.2f}" if numeric_columns[col].get('mean') is not None else 'N/A'}</td>
                         <td>{f"{numeric_columns[col].get('median', 'N/A'):.2f}" if numeric_columns[col].get('median') is not None else 'N/A'}</td>
                         <td>{f"{numeric_columns[col].get('std', 'N/A'):.2f}" if numeric_columns[col].get('std') is not None else 'N/A'}</td>
                         <td>{f"{numeric_columns[col].get('min', 'N/A'):.2f}" if numeric_columns[col].get('min') is not None else 'N/A'}</td>
                         <td>{f"{numeric_columns[col].get('max', 'N/A'):.2f}" if numeric_columns[col].get('max') is not None else 'N/A'}</td>
                            <td>{numeric_columns[col].get('unique_count', 0)}</td>
                            <td>{numeric_columns[col].get('missing_count', 0)}</td>
                        </tr>
                        ''' for col in numeric_columns])}
                    </table>
                    ''' if numeric_columns else '<p>No numeric columns found.</p>'}
                </div>
                
                <div class="section">
                    <h2>Categorical Columns Analysis</h2>
                    {f'''
                    <table class="table categorical">
                        <tr>
                            <th>Column</th>
                            <th>Unique Count</th>
                            <th>Missing Count</th>
                            <th>Top Values</th>
                        </tr>
                        {''.join([f'''
                        <tr>
                            <td>{col}</td>
                            <td>{categorical_columns[col].get('unique_count', 0)}</td>
                            <td>{categorical_columns[col].get('missing_count', 0)}</td>
                            <td>{', '.join([f'{k}: {v}' for k, v in list(categorical_columns[col].get('top_values', {}).items())[:3]])}</td>
                        </tr>
                        ''' for col in categorical_columns])}
                    </table>
                    ''' if categorical_columns else '<p>No categorical columns found.</p>'}
                </div>
            </body>
            </html>
            """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
    except Exception as e:
        print(f"Error saving HTML report: {e}")
        # Create a simple error report
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Generating Report</h1>
            <p>Failed to generate EDA report: {str(e)}</p>
        </body>
        </html>
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(error_html)


def generate_pdf_report(summary: str, df: pd.DataFrame, steps: list, logs: list = None) -> str:
    """Generate comprehensive PDF report with EDA visualizations, logs, and Gemini-generated cover image"""
    try:
        pdf_path = "temp_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # =====================
        # GEMINI COVER IMAGE
        # =====================
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = """
            Generate a professional, modern cover illustration for a Data Preprocessing and Exploratory Data Analysis (EDA) report.
            Theme: Data science, charts, analytics, AI, clean minimalistic style.
            """
            # ⚠️ This is experimental, correct call is generate_images
            result = model.generate_images(prompt=prompt, size="1024x1024")

            if hasattr(result, "images") and len(result.images) > 0:
                img_bytes = io.BytesIO(result.images[0])
                story.append(Image(img_bytes, width=500, height=350))
                story.append(Spacer(1, 20))
        except Exception as img_err:
            print("Gemini image generation failed:", img_err)

        # =====================
        # TITLE + METADATA
        # =====================
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("Data Preprocessing & EDA Report", title_style))
        story.append(Spacer(1, 20))

        story.append(Paragraph(f"<b>Generated on:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Total Processing Steps:</b> {len(steps)}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # =====================
        # DATASET OVERVIEW
        # =====================
        story.append(Paragraph("Dataset Overview", styles["Heading2"]))
        story.append(Paragraph(f"<b>Shape:</b> {df.shape[0]} rows × {df.shape[1]} columns", styles["Normal"]))
        story.append(Paragraph(f"<b>Memory Usage:</b> {df.memory_usage(deep=True).sum() / 1024:.2f} KB", styles["Normal"]))
        story.append(Spacer(1, 20))

        # =====================
        # EDA VISUALIZATIONS
        # =====================
        

        # 2. Correlation heatmap (numeric only)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 1:
            plt.figure(figsize=(6, 5))
            sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            story.append(Image(buf, width=400, height=300))
            plt.close()

        # 3. Histograms for first few numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns[:3]
        for col in num_cols:
            plt.figure(figsize=(5, 3))
            sns.histplot(df[col].dropna(), kde=True, bins=20)
            plt.title(f"Distribution of {col}")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            story.append(Image(buf, width=350, height=250))
            plt.close()

        story.append(Spacer(1, 20))

        # =====================
        # PROCESSING STEPS
        # =====================
        story.append(Paragraph("Processing Steps & Audit Trail", styles["Heading2"]))
        for i, step in enumerate(steps, 1):
            story.append(Paragraph(f"<b>{i}.</b> {step}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # =====================
        # SYSTEM LOGS
        # =====================
        if logs:
            story.append(Paragraph("System Logs", styles["Heading2"]))
            for log in logs[-20:]:
                story.append(Paragraph(log, styles["Normal"]))
            story.append(Spacer(1, 20))

        # =====================
        # AI SUMMARY (Fix)
        # =====================
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            ai_summary = model.generate_content(f"Summarize this dataset with shape {df.shape}, columns {list(df.columns)}")
            story.append(Paragraph("AI-Generated Summary", styles["Heading2"]))
            story.append(Paragraph(ai_summary.text, styles["Normal"]))
            story.append(Spacer(1, 20))
        except Exception as sum_err:
            print("AI summary generation failed:", sum_err)

        # =====================
        # RECOMMENDATIONS
        # =====================
        story.append(Paragraph("Recommendations & Next Steps", styles["Heading2"]))
        recommendations = [
            "• Review column distributions for anomalies",
            "• Consider imputing missing values or dropping high-null columns",
            "• Apply outlier detection methods on numeric features",
            "• Perform feature engineering for categorical variables",
            "• Document transformations for reproducibility"
        ]
        for rec in recommendations:
            story.append(Paragraph(rec, styles["Normal"]))

        # =====================
        # BUILD PDF
        # =====================
        doc.build(story)
        return pdf_path

    except Exception as e:
        print("PDF generation error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")    
    
    if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app, host="0.0.0.0", port=8000)
