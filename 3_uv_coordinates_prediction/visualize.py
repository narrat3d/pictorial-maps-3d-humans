import math
from PIL import Image
from mathutils import Color
import numpy as np
from data_loader import load_depth_map, load_uv_map
from config import RGB_COLORS


def calculate_uv_map(arr):
    def uv2rgb(uv):
        [u, v] = uv
        
        if (u == 0 and v == 0):
            return [0., 0., 0.]
        
        u_norm = u - 0.5
        v_norm = v - 0.5
        
        angle = math.atan2(v_norm, u_norm) + math.pi
        hue = angle / (2 * math.pi)
    
        # source: https://stackoverflow.com/questions/13211595/how-can-i-convert-coordinates-on-a-circle-to-coordinates-on-a-square
        circle_u = u_norm * math.sqrt(1 - 0.5 * v_norm**2)
        circle_v = v_norm * math.sqrt(1 - 0.5 * u_norm**2)
        
        length = math.sqrt(circle_u**2 + circle_v**2)
        saturation = length * 2
        
        c = Color()
        c.hsv = (hue, saturation, 1.0)
        
        return [c.r, c.g, c.b]
    
    uv_map_colored = np.apply_along_axis(uv2rgb, 2, arr)

    img = Image.fromarray(np.uint8(np.swapaxes(np.flip(uv_map_colored * 255, 1), 0, 1)))
    
    return img


def show_depth_map(path):
    arr, _ = load_depth_map(path)
    arr = np.squeeze(arr)
    
    img = Image.fromarray(np.uint8(np.swapaxes(np.flip(arr, 1), 0, 1) * 255))
    img.show()
    
    return img


def show_body_parts_mask(path):
    body_parts_image = Image.open(path)
    body_parts_image = body_parts_image.getchannel(0)
    
    body_parts_image_np = np.expand_dims(np.asarray(body_parts_image), -1)
    
    def body_part2rgb(body_part_index):
        if (body_part_index < 255):
            index = body_part_index[0]
            return np.array(RGB_COLORS[index]) * 255
        else :
            return [0, 0, 0]
    
    body_parts_image_np_colored = np.apply_along_axis(body_part2rgb, 2, body_parts_image_np)
    
    img = Image.fromarray(np.uint8(np.squeeze(body_parts_image_np_colored)))
    img.show()
    
    return img
    
    
def show_uv_map(path):
    uv_map = load_uv_map(path, 256)[0]
    
    img = calculate_uv_map(uv_map[..., 1:])
    img.show()
    
    return img


if __name__ == '__main__':
    # img = show_depth_map(r"C:\Users\raimund\Downloads\tmp2\rp_caren_posed_017_0_0_female_large\depth_right.npz")
    img = show_uv_map(r"C:\Users\raimund\Downloads\tmp2\rp_stephen_posed_037_0_0_male_small\uv_front.npz")
    img.save(r"C:\Users\raimund\Downloads\uv_front.png")