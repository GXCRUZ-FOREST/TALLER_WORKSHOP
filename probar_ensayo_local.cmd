@echo off
setlocal
set "PROYECTO=C:\Users\LENOVO\Documents\Codex\2026-11-10\vamso-a-hacer-c-peque-a"
set "PYTHONPATH=%PROYECTO%\work\deps"
python "%PROYECTO%\outputs\cnn_frijol.py" predecir --imagen "C:\Users\LENOVO\Downloads\beans128\1295_test_healthy.jpg"
pause

