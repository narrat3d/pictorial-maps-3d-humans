from model_parts import get_inputs, encode_coords, encode_image, get_coords_shape,\
    get_local_features_from_image, split_coords, decode_features
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Add


def disn_two_stream(num_pose_points):
    [image_2d, sdf_coords, sdf_coords_normalized, _] = get_inputs(num_pose_points)
    batch_size, num_samples = get_coords_shape(sdf_coords)
    
    [sdf_coords_xy, _, _] = split_coords(sdf_coords)
    sdf_coords_encoded = encode_coords(sdf_coords_normalized)
    
    global_features, feature_maps = encode_image(image_2d, batch_size)
    global_features_repeated = tf.repeat(global_features, num_samples, axis=1)
    
    local_features = get_local_features_from_image(feature_maps, sdf_coords_xy, batch_size, num_samples)

    local_features_branch = Concatenate()([sdf_coords_encoded, local_features])
    local_features_branch = decode_features(local_features_branch)
    
    global_features_branch = Concatenate()([sdf_coords_encoded, global_features_repeated])
    global_features_branch = decode_features(global_features_branch)
    
    sdf = Add(name="sdf3d")([local_features_branch, global_features_branch])
    return [image_2d, sdf_coords], [sdf]


def disn_two_stream_pose(num_pose_points):
    [image_2d, sdf_coords, sdf_coords_normalized, pose_coords] = get_inputs(num_pose_points)
    batch_size, num_samples = get_coords_shape(sdf_coords)
    
    [sdf_coords_xy, _, _] = split_coords(sdf_coords)
    sdf_coords_encoded = encode_coords(sdf_coords_normalized)
    
    pose_coords_encoded = encode_coords(pose_coords)
    pose_coords_encoded = tf.reshape(pose_coords_encoded, (-1, 1, num_pose_points * pose_coords_encoded.shape[2]))
    pose_coords_repeated = tf.repeat(pose_coords_encoded, num_samples, 1) 
    
    global_features, feature_maps = encode_image(image_2d, batch_size)
    global_features_repeated = tf.repeat(global_features, num_samples, axis=1)
    
    local_features = get_local_features_from_image(feature_maps, sdf_coords_xy, batch_size, num_samples)

    local_features_branch = Concatenate()([sdf_coords_encoded, pose_coords_repeated, local_features])
    local_features_branch = decode_features(local_features_branch)
    
    global_features_branch = Concatenate()([sdf_coords_encoded, pose_coords_repeated, global_features_repeated])
    global_features_branch = decode_features(global_features_branch)
    
    sdf = Add(name="sdf3d")([local_features_branch, global_features_branch])
    return [image_2d, sdf_coords, pose_coords], [sdf]


def disn_one_stream(num_pose_points):
    [image_2d, sdf_coords, sdf_coords_normalized, _] = get_inputs(num_pose_points)
    
    batch_size, num_samples = get_coords_shape(sdf_coords)
    
    [sdf_coords_xy, _, _] = split_coords(sdf_coords)
    sdf_coords_encoded = encode_coords(sdf_coords_normalized)
    
    global_features, feature_maps = encode_image(image_2d, batch_size)
    global_features_repeated = tf.repeat(global_features, num_samples, axis=1)
    
    local_features = get_local_features_from_image(feature_maps, sdf_coords_xy, batch_size, num_samples)

    merged_features = Concatenate()([sdf_coords_encoded, global_features_repeated, local_features])

    sdf = decode_features(merged_features, name="sdf3d")
    return [image_2d, sdf_coords], [sdf]


def disn_one_stream_pose(num_pose_points):
    [image_2d, sdf_coords, sdf_coords_normalized, pose_coords] = get_inputs(num_pose_points)
    
    batch_size, num_samples = get_coords_shape(sdf_coords)
    
    [sdf_coords_xy, _, _] = split_coords(sdf_coords)
    sdf_coords_encoded = encode_coords(sdf_coords_normalized)
    
    pose_coords_encoded = encode_coords(pose_coords)
    pose_coords_encoded = tf.reshape(pose_coords_encoded, (-1, 1, num_pose_points * pose_coords_encoded.shape[2]))
    pose_coords_repeated = tf.repeat(pose_coords_encoded, num_samples, 1) 
    
    global_features, feature_maps = encode_image(image_2d, batch_size)
    global_features_repeated = tf.repeat(global_features, num_samples, axis=1)
    
    local_features = get_local_features_from_image(feature_maps, sdf_coords_xy, batch_size, num_samples)

    merged_features = Concatenate()([sdf_coords_encoded, pose_coords_repeated, global_features_repeated, local_features]) # 

    sdf = decode_features(merged_features, name="sdf3d")
    return [image_2d, sdf_coords, pose_coords], [sdf]


BUILDS = {
    "disn_two_stream": disn_two_stream,
    "disn_two_stream_pose": disn_two_stream_pose,
    "disn_one_stream": disn_one_stream,
    "disn_one_stream_pose": disn_one_stream_pose
}