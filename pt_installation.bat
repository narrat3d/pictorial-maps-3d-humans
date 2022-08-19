set PYTHON_DIR=C:\Python385
set VENV_DIR=%userprofile%\VirtualEnvs\PT
%PYTHON_DIR%\Scripts\pip install virtualenv

set PYTHONHOME=
set PYTHONPATH=%PYTHON_DIR%\Lib;%PYTHON_DIR%\Lib\site-packages;
%PYTHON_DIR%\Scripts\virtualenv.exe -p %PYTHON_DIR%\python.exe %VENV_DIR%

set PYTHONPATH=
call %VENV_DIR%\Scripts\activate.bat
pip install -r pt_requirements.txt