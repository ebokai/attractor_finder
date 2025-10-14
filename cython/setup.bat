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
echo [1/3] Building renderer...
python setup_renderer.py build_ext --inplace --compiler=mingw32
if errorlevel 1 (
    echo ERROR: Failed to build renderer.pyx
    exit /b 1
)
echo.

REM Build iterator extension
echo [2/3] Building batch renderer...
python setup_batch_renderer.py build_ext --inplace --compiler=mingw32
if errorlevel 1 (
    echo ERROR: Failed to build batch_renderer.pyx
    exit /b 1
)
echo.

REM Build iterator extension
echo [3/3] Building iterator...
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
