# Rendering tool for pictorial 3D humans

This is a Python-based ray marcher which is able to render SDFs grids with textures.
An introductory description can be found here: http://osgl.ethz.ch/training/Story_Raymarcher_in_Python_II.pdf

## Configurations

* parts = shows the body parts in different colors
* depth = create a depth map and body part mask image
* render = applies the generated textures
* points = creates a point cloud file, which is converted to a x3d mesh
* video = render a video from 360° keyframes by FFmpeg

## Notes

* Increase the scaling_factor to close some gaps between the body parts.