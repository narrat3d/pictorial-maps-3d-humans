:: directory of the checked-out repository (i.e., where this *.bat file is located)
set REPOSITORY_FOLDER=E:\Repositories\pictorial-maps-3d-humans
cd /D %REPOSITORY_FOLDER%

:: folder where the trained models are stored
set MODEL_FOLDER=%cd%\models

:: annotations from supervise.ly
set FIGURE_TARFILE=%userprofile%\Downloads\PictorialFigures.tar
:: folder where temporary files are stored during processing
set FIGURE_TMPFOLDER=%cd%\data\tmp
:: folder where files (e.g., skeletons, SDFs, UV coordinates, textures) are stored for each processing stage 
set FIGURE_OUTFOLDER=%cd%\data\out
:: optionally add image names (without extension, without quotes, separated by |) which shall be processed exclusively
set FIGURE_SUBFOLDERS=""

set ORIGINAL_PATH=%PATH%

set PYTHONHOME=%userprofile%\VirtualEnvs\TF2
set PYTHONPATH=%PYTHONHOME%\Lib\site-packages;%cd%\0_data_preparation
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
set PATH=%PYTHONHOME%\scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%ORIGINAL_PATH%

call activate.bat
python %cd%\0_data_preparation\supervisely_to_narrat3d_characters.py --supervisely_tar_file=%FIGURE_TARFILE% --tmp_folder=%FIGURE_TMPFOLDER% --sub_folders=%FIGURE_SUBFOLDERS% 
python %cd%\0_data_preparation\supervisely_extract_characters.py --tmp_folder=%FIGURE_TMPFOLDER% --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
call deactivate

set PYTHONHOME=%userprofile%\VirtualEnvs\TF1
set PYTHONPATH=%PYTHONHOME%\Lib\site-packages;%cd%\1_3d_pose_estimation
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set PATH=%PYTHONHOME%\scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%ORIGINAL_PATH%

call activate.bat
python %cd%\1_3d_pose_estimation\src\predict_3dpose.py --model_folder=%MODEL_FOLDER%\1_3d_pose_estimation --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
call deactivate

set PYTHONHOME=%userprofile%\VirtualEnvs\TF2
set PYTHONPATH=%PYTHONHOME%\Lib\site-packages;%cd%\0_data_preparation;%cd%\2_3d_body_part_inference;%cd%\3_uv_coordinates_prediction;%cd%\5_ray_marching
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
set PATH=%PYTHONHOME%\scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%ORIGINAL_PATH%

call activate.bat
python %cd%\0_data_preparation\supervisely_extract_body_parts.py --model_folder=%MODEL_FOLDER%\0_data_preparation\scale_predictors --out_folder=%FIGURE_OUTFOLDER%  --sub_folders=%FIGURE_SUBFOLDERS%
python %cd%\2_3d_body_part_inference\inference.py --model_folder=%MODEL_FOLDER%\2_3d_body_part_inference --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
python %cd%\5_ray_marching\raymarching_numba_body.py --mode=depth --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
python %cd%\3_uv_coordinates_prediction\inference.py --model_folder=%MODEL_FOLDER%\3_uv_coordinates_prediction --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
python %cd%\0_data_preparation\texture_inpainting.py --out_folder=%FIGURE_OUTFOLDER% --tmp_folder=%FIGURE_TMPFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
call deactivate

set PYTHONHOME=%userprofile%\VirtualEnvs\PT
set PYTHONPATH=%PYTHONHOME%\Lib\site-packages;%cd%\4_texture_inpainting
set PATH=%PYTHONHOME%\scripts;%ORIGINAL_PATH%

call activate.bat
python %cd%\4_texture_inpainting\infer_sample.py --model_folder=%MODEL_FOLDER%\4_texture_inpainting --tmp_folder=%FIGURE_TMPFOLDER% --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
call deactivate


set PYTHONHOME=%userprofile%\VirtualEnvs\TF2
set PYTHONPATH=%PYTHONHOME%\Lib\site-packages;%cd%\5_ray_marching
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
set PATH=%PYTHONHOME%\scripts;%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%ORIGINAL_PATH%

call activate.bat
:: display the constructed figure
python %cd%\5_ray_marching\raymarching_numba_body.py --mode=render --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
:: optionally export figure as a point cloud and convert to a mesh
python %cd%\5_ray_marching\raymarching_numba_body.py --mode=points --out_folder=%FIGURE_OUTFOLDER% --sub_folders=%FIGURE_SUBFOLDERS%
call deactivate.bat

set PYTHONPATH=
set BLENDERHOME=C:\Program Files\Blender Foundation\Blender 3.1
set PATH=%BLENDERHOME%;%ORIGINAL_PATH%

:: optionally convert figure to a .glb mesh
blender %cd%\0_data_preparation\x3d_to_gltf.blend -P %cd%\0_data_preparation\x3d_to_gltf.py -b -- %FIGURE_OUTFOLDER% %FIGURE_SUBFOLDERS%