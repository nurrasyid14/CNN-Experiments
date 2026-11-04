@echo off
setlocal
cd /d "%~dp0.."

if "%CNN_EPOCHS%"=="" set "CNN_EPOCHS=10"
if "%CNN_BATCH_SIZE%"=="" set "CNN_BATCH_SIZE=32"

set "PYTHON=py -3"
if not "%CNN_PYTHON%"=="" set "PYTHON=%CNN_PYTHON%"

echo [1/5] Configure CMake
cmake --preset windows-release || exit /b %errorlevel%

echo [2/5] Build CMake targets
cmake --build build --preset windows-release || exit /b %errorlevel%

echo [3/5] Prepare both datasets
%PYTHON% scripts\prepare_datasets.py || exit /b %errorlevel%

echo [4/5] Train SIMPLIcity
%PYTHON% scripts\train_cnn.py --experiment simplicity --epochs %CNN_EPOCHS% --batch-size %CNN_BATCH_SIZE% || exit /b %errorlevel%
%PYTHON% scripts\analyze_results.py --experiment simplicity || exit /b %errorlevel%

echo [5/5] Train Aerial
%PYTHON% scripts\train_cnn.py --experiment aerial --epochs %CNN_EPOCHS% --batch-size %CNN_BATCH_SIZE% || exit /b %errorlevel%
%PYTHON% scripts\analyze_results.py --experiment aerial || exit /b %errorlevel%

echo Pipeline complete.
echo Checkpoints: outputs\checkpoints\experiments\
echo Reports:     outputs\reports\
endlocal
