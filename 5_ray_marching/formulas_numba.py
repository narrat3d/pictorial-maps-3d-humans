from numba import cuda
import math

@cuda.jit(device=True)
def cuboid(point, dimensions):
    # adapted from https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm 
    # Box - exact
    q = sub2(abs2(point), dimensions)
    [qx, qy, qz] = q
    
    return length2(max2(q, 0.0)) + min(max(qx, max(qy, qz)), 0.0)


@cuda.jit(device=True)
def sphere(point, r):
    return length2(point) - r


@cuda.jit(device=True)
def union(d1, d2):
    return min(d1, d2)


@cuda.jit(device=True)
def difference(d1, d2):
    return max(d1, -d2);

# source: https://www.iquilezles.org/www/articles/smin/smin.htm > polynomial smooth min
# https://www.shadertoy.com/view/lt3BW2
@cuda.jit(device=True)
def union_smooth(d1, d2, k):
    h = max(k - abs(d1 - d2), 0.0) 
    return min(d1, d2) - h*h*0.25/k


# level set
@cuda.jit(device=True)
def cuboid2(x, y, z, dx, dy, dz):
    return 0.25*abs(abs(0.25*x + y) + abs(0.25*x - y) + 2*z) + 0.25*abs(abs(0.25*x + y) + abs(0.25*x - y) - 2*z) - 64

@cuda.jit(device=True)
def cuboid3(x, y, z, dx, dy, dz):
    return max(abs(0.25*x), abs(y), abs(z)) - 64


# adapted from https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
@cuda.jit(device=True)
def add2(vector1, vector2):
    vector = (
        vector1[0] + vector2[0],
        vector1[1] + vector2[1],
        vector1[2] + vector2[2],
    )
    return vector


@cuda.jit(device=True)
def sub2(vector2, vector1):
    vector = (
        vector2[0] - vector1[0],
        vector2[1] - vector1[1],
        vector2[2] - vector1[2],
    )
    return vector
    
    
@cuda.jit(device=True)
def mult2(vector, scalar):
    vector = (
        vector[0] * scalar,
        vector[1] * scalar,
        vector[2] * scalar
    )
    return vector


@cuda.jit(device=True)
def abs2(vector):
    vector = (
        abs(vector[0]),
        abs(vector[1]),
        abs(vector[2]),
    )
    return vector


@cuda.jit(device=True)
def max2(vector, number):
    vector = (
        max(vector[0], number),
        max(vector[1], number),
        max(vector[2], number),
    )
    return vector

@cuda.jit(device=True)
def min2(vector, number):
    vector = (
        min(vector[0], number),
        min(vector[1], number),
        min(vector[2], number),
    )
    return vector


@cuda.jit(device=True)
def dot2(vector1, vector2):
    n = vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]

    return n


@cuda.jit(device=True)
def length2(vector):
    n = math.sqrt(dot2(vector, vector))

    return n


@cuda.jit(device=True)
def normalize2(vector):
    n = length2(vector)
    normalized_vector = mult2(vector, 1 / n)
    
    return normalized_vector


@cuda.jit(device=True)
def tx(point, matrix):
    [x, y, z] = point
    
    vector = (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    )
    return vector


@cuda.jit(device=True)
def translate(point, vector):
    return sub2(point, vector)


@cuda.jit(device=True)
def rotate_x(point, angle):
    c = math.cos(angle)
    s = math.sin(angle)

    rotation_matrix = (
        (1, 0, 0, 0),
        (0, c, -s, 0),
        (0, s, c, 0),
        (0, 0, 0, 1)
    )
    
    return tx(point, rotation_matrix)

@cuda.jit(device=True)
def rotate_y(point, angle):
    c = math.cos(angle)
    s = math.sin(angle)
        
    rotation_matrix = (
        (c, 0, s, 0),
        (0, 1, 0, 0),
        (-s, 0, c, 0),
        (0, 0, 0, 1)
    )
    
    return tx(point, rotation_matrix)

@cuda.jit(device=True)
def rotate_z(point, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    
    rotation_matrix = (
        (c, -s, 0, 0),
        (s, c, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1)
    )
    
    return tx(point, rotation_matrix)
