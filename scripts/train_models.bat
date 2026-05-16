@echo off
echo Starting Weekly ML Model Training...

:: %~dp0 resolves to the folder this .bat file lives in (the scripts/ folder)
:: Navigate one level up to the project root
cd /d "%~dp0.."

:: Use 'python' from your activated virtual environment or system PATH
echo Training Fictometer Model...
python "fictometer\scripts\trainmodel.py"

echo Training Spam Classifier Models...
python "spamclassifier\scripts\trainmodel.py"

echo Training Category Predictor (Word2Vec + ANN)...
python "categorypredictor\scripts\trainmodel_word2vec.py"

echo Training Category Predictor (ELMo + ANN)...
python "categorypredictor\scripts\trainmodel_elmo.py"

echo Training Complete.
