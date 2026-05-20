@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Machine Learning Pipeline Runner
REM NEW: Can trigger via FastAPI OR run standalone
REM Enhanced Windows error handling
REM ========================================

REM ===== CONFIGURATION =====
set "PROJECT_DIR=C:\Users\Administrator\Desktop\CUSTOMIZED PORTAL\machine-model"
set "FASTAPI_URL=http://127.0.0.1:8000"
set "USE_FASTAPI=true"
set "API_PORT=8000"

REM ===== SET PROJECT DIRECTORY =====
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Cannot change to project directory: %PROJECT_DIR%
    pause
    exit /b 1
)

REM ===== ACTIVATE VIRTUAL ENVIRONMENT =====
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM ===== CREATE TIMESTAMP =====
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set "TIMESTAMP=%mydate%_%mytime%"

REM ===== CREATE LOGS DIRECTORY =====
if not exist "logs" mkdir logs

echo.
echo ========================================
echo ML Pipeline started at %DATE% %TIME%
echo ========================================
echo Mode: %USE_FASTAPI%
echo.

REM ===== TRACK RESULTS =====
set "FORECAST_STATUS=PENDING"
set "ANOMALY_STATUS=PENDING"
set "FASTAPI_STATUS=PENDING"

REM ===== OPTION 1: START FASTAPI SERVER & TRIGGER VIA API =====
if "%USE_FASTAPI%"=="true" (
    echo.
    echo ========================================
    echo Starting FastAPI Server
    echo ========================================
    echo.
    
    REM Check if port is already in use
    echo Checking if port %API_PORT% is available...
    netstat -ano | findstr ":%API_PORT% " >nul
    
    if errorlevel 1 (
        echo ✅ Port %API_PORT% is available
        
        REM Start FastAPI in background (separate process)
        echo Starting FastAPI server...
        set "FASTAPI_LOG=logs\fastapi_%TIMESTAMP%.log"
        
        start "FastAPI Server" python -m uvicorn main:app --host 127.0.0.1 --port %API_PORT% --reload >> "!FASTAPI_LOG!" 2>&1
        
        REM Wait for server to start
        echo Waiting for FastAPI to initialize...
        timeout /t 5 /nobreak
        
        REM Check if server is running
        curl -s -o nul -w "%%{http_code}" http://127.0.0.1:%API_PORT%/health > temp_status.txt
        set /p HTTP_STATUS=<temp_status.txt
        del temp_status.txt
        
        if "!HTTP_STATUS!"=="200" (
            echo ✅ FastAPI server is running successfully
            set "FASTAPI_STATUS=RUNNING"
            goto :USE_API
        ) else (
            echo ⚠️ FastAPI failed to start - check logs
            set "FASTAPI_STATUS=FAILED"
            echo !HTTP_STATUS!
            goto :USE_SCRIPTS
        )
    ) else (
        echo ❌ Port %API_PORT% is already in use
        echo Please close the existing process or change API_PORT in this script
        echo Use: netstat -ano | findstr ":%API_PORT%" to find the process
        set "FASTAPI_STATUS=PORT_IN_USE"
        goto :USE_SCRIPTS
    )
)

REM ===== OPTION 1A: TRIGGER VIA FASTAPI ENDPOINTS =====
:USE_API
echo.
echo ========================================
echo Triggering training via FastAPI API
echo ========================================
echo.

REM Trigger anomaly training
echo [1/2] Triggering anomaly detection training...
set "ANOMALY_LOG=logs\anomaly_api_%TIMESTAMP%.log"

curl -X POST http://127.0.0.1:%API_PORT%/api/train/anomaly ^
  -H "Content-Type: application/json" ^
  > "!ANOMALY_LOG!" 2>&1

if errorlevel 1 (
    echo WARNING: Anomaly training API call FAILED
    set "ANOMALY_STATUS=FAILED"
) else (
    echo SUCCESS: Anomaly training triggered
    set "ANOMALY_STATUS=SUCCESS"
)

echo.

REM Trigger forecast training
echo [2/2] Triggering forecast training...
set "FORECAST_LOG=logs\forecast_api_%TIMESTAMP%.log"

curl -X POST http://127.0.0.1:%API_PORT%/api/train/forecast ^
  -H "Content-Type: application/json" ^
  > "!FORECAST_LOG!" 2>&1

if errorlevel 1 (
    echo WARNING: Forecast training API call FAILED
    set "FORECAST_STATUS=FAILED"
) else (
    echo SUCCESS: Forecast training triggered
    set "FORECAST_STATUS=SUCCESS"
)

REM Wait for background tasks
echo.
echo Waiting for background training tasks to complete...
timeout /t 10 /nobreak

goto :SUMMARY

REM ===== OPTION 2: RUN SCRIPTS DIRECTLY (FALLBACK) =====
:USE_SCRIPTS
echo.
echo ========================================
echo Running Python scripts directly (Fallback)
echo ========================================
echo.

REM STEP 1: Run anomaly detection
echo [1/2] Running anomaly_detector.py...
set "ANOMALY_LOG=logs\anomaly_detector_%TIMESTAMP%.log"

python models/anomaly_detector.py > "!ANOMALY_LOG!" 2>&1

if errorlevel 1 (
    echo WARNING: anomaly_detector.py FAILED - Check !ANOMALY_LOG!
    set "ANOMALY_STATUS=FAILED"
) else (
    echo SUCCESS: anomaly_detector.py completed
    set "ANOMALY_STATUS=SUCCESS"
)

echo.

REM STEP 2: Run cash-in forecast
echo [2/2] Running forecast_model.py...
set "FORECAST_LOG=logs\forecast_model_%TIMESTAMP%.log"

python models/forecast_model.py > "!FORECAST_LOG!" 2>&1

if errorlevel 1 (
    echo WARNING: forecast_model.py FAILED - Check !FORECAST_LOG!
    set "FORECAST_STATUS=FAILED"
) else (
    echo SUCCESS: forecast_model.py completed
    set "FORECAST_STATUS=SUCCESS"
)

REM ===== SUMMARY =====
:SUMMARY
echo.
echo ========================================
echo Pipeline Summary
echo ========================================
echo FastAPI Server:          %FASTAPI_STATUS%
echo Anomaly Detection:       %ANOMALY_STATUS%
echo Forecast Training:       %FORECAST_STATUS%
echo Completed at:            %TIME%
echo ========================================
echo.

if exist "!ANOMALY_LOG!" (
    echo Anomaly log:  !ANOMALY_LOG!
)
if exist "!FORECAST_LOG!" (
    echo Forecast log: !FORECAST_LOG!
)
echo.

REM ===== CHECK MODEL STATUS (if using FastAPI) =====
if "%FASTAPI_STATUS%"=="RUNNING" (
    echo Checking model status...
    curl -s http://127.0.0.1:%API_PORT%/api/models/status
    echo.
    echo.
)

REM ===== EXIT CODES =====
if "%FORECAST_STATUS%"=="FAILED" if "%ANOMALY_STATUS%"=="FAILED" (
    echo ❌ CRITICAL: Both trainings failed
    exit /b 2
)
if "%FORECAST_STATUS%"=="FAILED" (
    echo ⚠️ WARNING: Forecast failed
    exit /b 1
)
if "%ANOMALY_STATUS%"=="FAILED" (
    echo ⚠️ WARNING: Anomaly detection failed
    exit /b 1
)

echo ✅ All operations completed successfully
echo.

REM Keep window open if run manually
if "%1" neq "scheduled" (
    echo Press any key to close...
    pause >nul
)

exit /b 0
