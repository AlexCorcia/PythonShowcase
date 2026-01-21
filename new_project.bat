@echo off

echo =========================
echo  NUEVO PROYECTO PYTHON
echo =========================
echo.

set /p nombre=Nombre del proyecto: 

echo.
echo Copiando template...
mkdir %nombre%
xcopy template\app %nombre%\app /E /I > nul
copy template\requirements.txt %nombre% > nul
copy template\.gitignore %nombre% > nul
copy template\README.md %nombre% > nul

echo.
echo Creando entorno virtual...
python -m venv %nombre%\venv

echo.
echo LISTO! Proyecto creado: %nombre%
echo.

echo Para usar:
echo cd %nombre%
echo venv\Scripts\activate
echo python -m app.main

pause
