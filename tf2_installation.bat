set PYTHON_DIR=C:\Python385
set VENV_DIR=%userprofile%\VirtualEnvs\TF2
%PYTHON_DIR%\Scripts\pip install virtualenv

set PYTHONHOME=
set PYTHONPATH=%PYTHON_DIR%\Lib;%PYTHON_DIR%\Lib\site-packages;
%PYTHON_DIR%\Scripts\virtualenv.exe -p %PYTHON_DIR%\python.exe %VENV_DIR%

set PYTHONPATH=
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp;%PATH%
call %VENV_DIR%\Scripts\activate.bat
pip install -r tf2_requirements.txt