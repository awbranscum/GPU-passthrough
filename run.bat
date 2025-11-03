@echo off
echo ===============================
echo Starting Ollama GPU Server
echo ===============================

REM Activate the virtual environment
call "%~dp0Scripts\activate"

REM Run the FastAPI server
uvicorn ollama_gpu_server:app --host 0.0.0.0 --port 11434

pause
