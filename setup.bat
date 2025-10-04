@echo off
setlocal

REM ---- Configuration ----
REM Ensure MinGW is on PATH
set PATH=C:\mingw64\bin;%PATH%

echo.
echo =====================================================
echo   Building Cython extensions with MinGW32
echo =====================================================
echo.

REM Clean previous builds (optional)
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build renderer extension
echo [1/2] Building renderer...
python setup_renderer.py build_ext --inplace --compiler=mingw32
if errorlevel 1 (
    echo ERROR: Failed to build renderer.pyx
    exit /b 1
)
echo.

REM Build iterator extension
echo [2/2] Building iterator...
python setup_iterator.py build_ext --inplace --compiler=mingw32
if errorlevel 1 (
    echo ERROR: Failed to build iterator.pyx
    exit /b 1
)
echo.

echo =====================================================
echo   Build completed successfully!
echo =====================================================

pause
endlocal
