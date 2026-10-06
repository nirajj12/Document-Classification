@echo off
cd /d "%~dp0"
echo Use the document-classifier Conda environment before launching.
python -m streamlit run src/app.py
pause
