@echo off
setlocal

echo.
echo =====================================================
echo   Running main.py
echo =====================================================
echo.

REM ---- Run your Python script ----
python ./main.py --render_iterates 50000000 --n_attractors 10 --alpha 0.025

echo.
echo =====================================================
echo   Script finished.
echo   (Press any key to close this window)
echo =====================================================
echo.

pause
endlocal
