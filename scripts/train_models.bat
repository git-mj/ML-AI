@echo off
echo Starting Weekly ML Model Training...

:: Change directory to the absolute path of the project workspace
cd /d "c:\Users\mjay2\Desktop\Learn\Excercises\ML-AI"

:: Execute the training script using the absolute path to your Python interpreter
echo Training Fictometer Model...
"C:\Python313\python.exe" "fictometer\scripts\trainmodel.py"

echo Training Spam Classifier Models...
"C:\Python313\python.exe" "spamclassifier\scripts\trainmodel.py"

echo Training Category Predictor (Word2Vec + ANN)...
"C:\Python313\python.exe" "categorypredictor\scripts\trainmodel_word2vec.py"

echo Training Category Predictor (ELMo + ANN)...
"C:\Python313\python.exe" "categorypredictor\scripts\trainmodel_elmo.py"

echo Training Complete.
