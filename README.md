# Inferring 3D human figures from pictorial maps 

This is code for the article [Inferring Implicit 3D Representations from Human Figures on Pictorial Maps](https://doi.org/10.1080/15230406.2023.2224063). Visit the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/) for more information or find further READMEs in the subfolders.

![Figure_on_map](https://github.com/narrat3d/pictorial-maps-3d-humans/assets/9949879/d1384809-1013-44d2-b36f-9c0e751fa3df)


## Git configuration

* Use 'git submodule update --init --recursive --force --remote' to check out the submodules.
* Use 'git submodule update --remote --merge' to update the submodules once they are checked-out.

## Installation (on Windows)

* Install [Python 3.7 and 3.8](https://www.python.org/downloads/)
* Install [CUDA Toolkit 10.0 and 11.2](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNNs](https://developer.nvidia.com/rdp/cudnn-download)
* Install [Blender 2.93 and 3.1](https://www.blender.org/download/), only needed for creating training data and converting 3D meshes 
* Run tf1_installation.bat (needed for 3D pose estimation) after editing the PYTHON_DIR variable 
* Run tf2_installation.bat (needed for data preparation, 3D body part inference, uv coordinates prediction) after editing the PYTHON_DIR variable
* Run pt_installation.bat (needed for texture inpainting) after editing the PYTHON_DIR variable

## Inference with pre-trained models

* Download our test data and the pre-trained models from the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/).
* Run 3d_reconstruction.bat after editing the REPOSITORY_FOLDER, MODEL_FOLDER, FIGURE_TARFILE variables. 
* Compare your results with ours from the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/).

## Training the networks

* Download our training and validation data from the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/).
* See README.md in each sub-network folder for more information

## Creation of training and validation data

* Download the [SMPL-X plugin for Blender](https://smpl-x.is.tue.mpg.de/), poses from [AGORA](https://agora.is.tue.mpg.de/), textures from [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) and [Multi-Garment](http://virtualhumans.mpi-inf.mpg.de/mgn/)
* Run 0_data_preparation/body_part_splitter_batch.py
* Run 0_data_preparation/obj_to_sdf_batch.py

## Creation of test data

* Import our extracted test data from the [project website](http://narrat3d.ethz.ch/3d-humans-from-pictorial-maps/) to supervise.ly with the Supervisely plugin.
* Upload your own images to the imported project and annotate your own pictorial figures with supervise.ly.
* Export the annotations as .json + images and download the .tar file.

## Notes

* Optionally enable [long folder paths](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later) (> 256 characters) if you get problems with the 3D pose estimation models

![Eating](https://github.com/narrat3d/pictorial-maps-3d-humans/assets/9949879/57d77c51-86bd-4caa-8280-6ad3a8cc3d5f)

## Citation

Please cite the following article when using this code:

```
@article{schnuerer2024inferring,
  author = {Raimund Schnürer, A. Cengiz Öztireli, Magnus Heitzler, René Sieber and Lorenz Hurni},
  title = {Inferring implicit 3D representations from human figures on pictorial maps},
  journal = {Cartography and Geographic Information Science},
  volume = {51},
  number = {1},
  pages = {97-113},
  year = {2024},
  doi = {10.1080/15230406.2023.2224063}
}
```

© 2022-2023 ETH Zurich, Raimund Schnürer
