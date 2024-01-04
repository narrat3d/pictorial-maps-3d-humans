# Animated ray-marched figures with speech bubbles in a virtual globe

This is an adapted version of the [Cesium library](https://cesiumjs.org/) from 2017 (MIT License, Copyright by Cesium Contributors).
The original README is [here](README_CESIUM.md).

Speech bubbles are created by the [HTMLBillboards](https://github.com/ThadThompson/CesiumHtmlBillboards) and 
[Bubbly](https://github.com/LeaVerou/bubbly) library.

## Installation

node.js version 8.17.0 is required to run this project.  
`npm install` to install dependencies

## Data

Copy the inferred figures into the folder `Apps/figures`.
To add transformation files of motion-captured animations, you need to run the 
script `export_36m_animation.py` from `1_3d_pose_estimation/src`.
Geodata and 3D models are not included due to licensing restrictions.

## Usage

`npm run start` to start the development server  
`npm run build-watch` to build the project (incl. shaders)

Open `http://localhost:8080/HelloWorld.html` in your browser to see the project.

You can edit the code in `Apps/HelloCesium.js` and `Source/Shaders/BillboardCollectionFS.glsl`

To show satellite imagery, get a token from the Bing Maps API and insert it into `HelloCesium.js`.
To enable the terrain model, collect a bearer token in Cesium Ion and insert it into `server.js`.

Note that the initial compilation of the shaders takes a long time. 
After caching the shaders, reloading the page is much faster.
