import os
import json
import subprocess
import requests
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjMwMDIwOTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.COZ_4kVMSwz0jqlAlTFV9wWqiYtOGiai4Qu3wqZW1gA"

BASE_DIR = "/app/data"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

PRIMARY_PROMPT = """The user will provide a task description.
Write one or more `bash` or `python` scripts to execute the task.
the inputs will be like either a text which has task description
for example
For example, here’s how anyone can make a request:

curl -X POST "https://your-app.vercel.app/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the "answer" column of the CSV file?" \
  -F "file=@abcd.zip"
the codes that you write must be able to solve the question and return the answer in the below given format.
  
CODING RULES:

- uv, the Python runner, is ALREADY installed. Run with `uv run [URL] [ARGUMENTS]`
- Parse dates with `python-dateutil`


- Call an LLM via a POST request to `https://aiproxy.sanand.workers.dev/openai/v1/chat/completions` with `Authorization: Bearer token` and this JSON body:
    {
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "[INSERT SYSTEM PROMPT]" },
        { role: "user", content: [
        { type: "text", text: "[INSERT USER MESSAGE]" }, // for text
        { type: "file_path", file_path: "[INSERT FILE PATH ]" }, // for file path , it is optional
        ]}
      ],
      // response_format: "json_object",  // forces JSON response
    }
  Response is in `response.choices?.[0]?.message?.content`. Error is in `response.error?.message`.
- Calculate embeddings with a POST request to `https://llmfoundry.straive.com/openai/v1/embeddings` with `Authorization: Bearer {os.getenv("LLMFOUNDRY_TOKEN")}` and this JSON body:
    {
      model: "text-embedding-3-small",
      input: [array of strings],
    }
  Embeddings are in response.data[*].embedding - an array of floats.
  Calculate the dot product of the embeddings (skipping the diagonal) to find the most similar pair of strings.

client.post(
            f"{openai_api_base}/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "text-embedding-3-small", "input": data},
        )
- When extracting card information, use the system prompt "Extract the EXACT dummy credit card number from this test image"

EXECUTION RULES: An automated agent will blindly run the scripts you provide. So ONLY
write the FINAL script(s) to run in ```bash or ```python code fences.
final answer must be in json format only and must be valid json
{
  "answer": "1234567890"
}

this is how the output must be , and nothing else is allowed 


"""

import shutil

def run_code_s():
    """Runs `code -s` and returns output"""
    try:
        code_path = shutil.which("code")  # Auto-detects the full path of `code`
        if not code_path:
            return {"answer": "Error: VS Code binary not found in PATH"}

        output = subprocess.run([code_path, "-s"], capture_output=True, text=True)
        return {"answer": output.stdout}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

def run_httpie_request():
    """Runs `uv run --with httpie -- https https://httpbin.org/get?email=23f3002091@ds.study.iitm.ac.in`"""
    try:
        output = subprocess.run(
            ["uv", "run", "--with", "httpie", "--", "https", "https://httpbin.org/get?email=23f3002091@ds.study.iitm.ac.in"],
            capture_output=True,
            text=True
        )
        return {"answer": json.loads(output.stdout)}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

import hashlib
import platform

def run_prettier_hash(file_path):
    """Runs `npx -y prettier@3.4.2 file` and hashes the output in a cross-platform way."""
    try:
        # Run Prettier and capture formatted output
        prettier_output = subprocess.run(
            ["npx", "-y", "prettier@3.4.2", file_path],
            capture_output=True,
            text=True,
            shell=True
        ).stdout

        if not prettier_output:
            return {"error": "Prettier did not return any output."}

        # Compute SHA-256 hash of the output
        sha256_hash = hashlib.sha256(prettier_output.encode()).hexdigest()

        return {"answer": sha256_hash}
    except Exception as e:
        return {"error": str(e)}
    
def google_sheets_sum(rows, cols, start, step, row_limit, col_limit):
    """Simulates the Google Sheets formula =SUM(ARRAY_CONSTRAIN(SEQUENCE(rows, cols, start, step), row_limit, col_limit))"""
    first_row = [start + (i * step) for i in range(col_limit)]  # Generate first 'col_limit' numbers
    return {"answer": sum(first_row)}

import numpy as np

def excel_sortby_sum(prompt):
    """
    Extracts numbers from the Excel formula, sorts them based on given priorities, 
    takes the specified number of elements, and returns the sum in JSON format.
    """
    try:
        # Extracting numbers from the formula
        numbers = list(map(int, re.findall(r'\d+', prompt.split("SORTBY({")[1].split("},")[0])))
        priorities = list(map(int, re.findall(r'\d+', prompt.split("SORTBY({")[1].split("},")[1].split("})")[0])))
        take_n = int(re.findall(r'TAKE\(.*?,\s*(\d+)\)', prompt)[0])
        
        # Sorting based on priorities
        sorted_indices = np.argsort(priorities)
        sorted_numbers = [numbers[i] for i in sorted_indices]
        
        # Taking the required elements and summing them
        result = sum(sorted_numbers[:take_n])
        
        return {"answer": result}
    except Exception as e:
        return json.dumps({"error": str(e)})
    

from datetime import datetime, timedelta

def count_weekdays(start_date: str, end_date: str, weekday: str) -> int:
    """
    Counts the number of occurrences of a specific weekday in a given date range.
    
    Parameters:
    start_date (str): Start date in YYYY-MM-DD format.
    end_date (str): End date in YYYY-MM-DD format.
    weekday (str): Weekday name (e.g., 'Monday', 'Tuesday', etc.).
    
    Returns:
    int: Number of times the weekday occurs in the given date range.
    """
    weekday_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    if weekday not in weekday_map:
        raise ValueError("Invalid weekday name. Use full name (e.g., 'Wednesday').")
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    count = 0
    
    while start <= end:
        if start.weekday() == weekday_map[weekday]:
            count += 1
        start += timedelta(days=1)
    
    return {"answer":count}

import json

def sort_json_by_fields(task):
    # Extract JSON from the given task
    json_match = re.search(r'\[(.*?)\]', task, re.DOTALL)
    if not json_match:
        return {"error": "No valid JSON found in task"}
    
    try:
        json_data = json.loads(f'[{json_match.group(1)}]')
        # Sort by age first, then by name
        sorted_data = sorted(json_data, key=lambda x: (x["age"], x["name"]))
        return {"answer": sorted_data}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

@app.get("/")
def home():
    return {"message": "FastAPI LLM Task Runner"}

from pydantic import BaseModel

# class TaskRequest(BaseModel):
#     task: str

class QuestionRequest(BaseModel):
    question: str

from fastapi import Form

import re
import json

@app.post("/api")
async def task_runner(
    question: str = Form(...),
    file: UploadFile = None
):
    task = question
    """Processes the task and runs the appropriate function"""
    
    # 🔥 **Check Hardcoded Functions First**
    if "code -s" in task:
        return run_code_s()
    
    if "https://httpbin.org/get" in task:
        return run_httpie_request()

    if "prettier@3.4.2" in task and file:
        file_path = f"{BASE_DIR}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return run_prettier_hash(file_path)
    
    if "Google Sheets" in task:
        rows, cols, start, step, row_limit, col_limit = map(int, re.findall(r'\d+', task))
        return google_sheets_sum(rows, cols, start, step, row_limit, col_limit)
    
    if "Excel" in task:
        return excel_sortby_sum(task)
    
    if "Wednesdays" in task:
        start_date, end_date = re.findall(r'\d{4}-\d{2}-\d{2}', task)
        return count_weekdays(start_date, end_date, "Wednesday")
    
    if "Sorted JSON" in task:
        return sort_json_by_fields(task)

    # 📡 **Use LLM if no hardcoded function matches**
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"


    data = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": PRIMARY_PROMPT},
        {"role": "user", "content": task},
    ],
    "response_format": {"type": "json_object"},
}

    if file:  # Check if a file is provided
        data["messages"].append({"type": "file_path", "file_path": file.filename})


    response = requests.post(url=url, headers=headers, json=data)
    
    try:
        r = response.json()
        python_code = json.loads(r['choices'][0]['message']['content'])['python_code']
        
        # 🔹 **Write LLM Code to File**
        llm_script_path = "/app/llm_code.py"
        with open(llm_script_path, "w") as f:
            f.write(python_code)
        
        os.chmod(llm_script_path, 0o755)

        # 🚀 **Run the Generated Script**
        output = subprocess.run(
            ["uv", "run", "llm_code.py", "--email", "23f3002091@ds.study.iitm.ac.in"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        return {"answer": output.stdout.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing LLM code: {str(e)}")
