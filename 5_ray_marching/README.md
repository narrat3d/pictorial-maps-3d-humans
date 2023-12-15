# Rendering tool for pictorial 3D humans

This is a Python-based ray marcher which is able to render SDFs grids with textures.
An introductory description can be found here: http://osgl.ethz.ch/training/Story_Raymarcher_in_Python_II.pdf

## Configurations

Run the file `raymarching_numba_body.py` to render a single figure.

* parts = shows the body parts in different colors
* depth = create a depth map and body part mask image
* render = applies the generated textures
* points = creates a point cloud file, which is converted to a x3d mesh
* video = render a video from 360° keyframes by FFmpeg

## Animations

1. Run `raymarching_numba_body_h36m_animation.py` to animate the figure 'nouvelle_image__312368' (eating) or 'Antoine_Corbineau_Metropolitan_Eurostar_Bruxelles_Brugman_map' (shopping). The required files 'skeleton_animation.npy' and 'rotation_animation.npy' are provided in the result data. To create your own transformation files of motion-captured animations, you need to run the script `export_36m_animation.py` from `1_3d_pose_estimation/src`.
2. Run `raymarching_numba_body_scripted_animation.py` to animate the figure 'airbnb' (walking) or '56937af294b81d9ec26ef94378efcf1d--maps-history-travel-illustration' (waving). These figures are included in the result data. To change the animations, you need to uncomment and comment code in the methods `shapes()` (i.e. point) and `my_kernel()` (i.e. rotated_ray_point, rotated_surface_normal).

## Notes

* Increase the scaling_factor to close some gaps between the body parts.