@echo off
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
pip freeze > requirements.lock.txt
echo.
echo Setup done. Next steps:
echo   ollama pull qwen2.5:1.5b
echo   .venv\Scripts\activate
echo   pytest
echo   uvicorn app.main:app --port 8000
goto :eof
:error
echo Setup failed.
exit /b 1