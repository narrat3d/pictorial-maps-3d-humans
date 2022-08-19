set PYTHON_DIR=C:\Python373
set VENV_DIR=%userprofile%\VirtualEnvs\TF1
%PYTHON_DIR%\Scripts\pip install virtualenv

set PYTHONHOME=
set PYTHONPATH=%PYTHON_DIR%\Lib;%PYTHON_DIR%\Lib\site-packages;
%PYTHON_DIR%\Scripts\virtualenv.exe -p %PYTHON_DIR%\python.exe %VENV_DIR%

set PYTHONPATH=
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;%PATH%
call %VENV_DIR%\Scripts\activate.bat
pip install -r tf1_requirements.txt