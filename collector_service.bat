@echo off
REM Script Windows pour lancer le collecteur externe en mode daemon
REM Usage: collector_service.bat [intervalle_en_minutes]

set INTERVAL=%1
if "%INTERVAL%"=="" set INTERVAL=1

echo [COLLECTOR-SERVICE] Demarrage du collecteur externe...
echo [COLLECTOR-SERVICE] Intervalle: %INTERVAL% minute(s)
echo.

python collector_service.py --daemon --interval %INTERVAL% --log-file logs/collector_service.log

pause

