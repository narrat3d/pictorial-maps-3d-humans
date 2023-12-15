#extension GL_EXT_frag_depth : enable

uniform vec3 cameraForward;
uniform vec3 cameraUp;
uniform vec3 cameraRight;
uniform float tanCameraFovHalf;
uniform float tanCameraFovYHalf;
uniform float depthA;
uniform float depthB;
uniform vec3 position;
uniform float ids[3];
uniform float selectedID;
uniform float animationTime;

//it's only possible to pass 16 textures to the shader
uniform sampler2D sdf_objects[14];
uniform sampler2D image_texture;
uniform float scales[14];
uniform float adjusted_body_size;

uniform sampler2D transformation_data;
//uniform float skeleton_points[64]; //21*3
//uniform float object_translations[64]; //14*3
//uniform float body_part_translations[64]; //14*3
//uniform float body_part_rotations[64]; //14*4

uniform vec3 model_rotation;
uniform vec3 headbang_rotation;

const float MINIFIER = 5. / 1.; //for picking test: 0.0001;
const float EPSILON = 0.1;
const int MAX_STEPS = 64;
const float PI = 3.141592654;
const float SQRT2 = 1.41421356;

const float body_size = 300.;
const float body_size_half = body_size / 2.;
const int body_part_size = 64;
const float body_part_size_half = float(body_part_size) / 2.;
const float body_part_texture_size = sqrt(float(body_part_size * body_part_size * body_part_size));



struct obj2 {
    float distance;
    int min_index;
};

float rayBox(vec3 p, vec3 b) {
    //return length(max(abs(p) - b, 0.));
    vec3 d = abs(p) - b;
    return max(d.x, max(d.y, d.z));
}

vec3 negate(vec3 p) {
    return vec3(-p.x, -p.y, -p.z);
}

vec3 translate(vec3 p, vec3 d) {
    return p - d;
}

vec3 translateZ(vec3 p, float dz) {
    return p - vec3(0, 0, dz);
}

mat4 Rot4X(float a ) {
    float c = cos( a );
    float s = sin( a );
    return mat4( 1, 0, 0, 0,
                 0, c,-s, 0,
                 0, s, c, 0,
                 0, 0, 0, 1 );
}

mat4 Rot4Y(float a ) {
    float c = cos( a );
    float s = sin( a );
    return mat4( c, 0, s, 0,
                 0, 1, 0, 0,
                -s, 0, c, 0,
                 0, 0, 0, 1 );
}

mat4 Rot4Z(float a ) {
    float c = cos( a );
    float s = sin( a );
    return mat4(
        c,-s, 0, 0,
        s, c, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
     );
}

mat4 transposeM4(in mat4 m ) {
    vec4 r0 = m[0];
    vec4 r1 = m[1];
    vec4 r2 = m[2];
    vec4 r3 = m[3];

    mat4 t = mat4(
         vec4( r0.x, r1.x, r2.x, r3.x ),
         vec4( r0.y, r1.y, r2.y, r3.y ),
         vec4( r0.z, r1.z, r2.z, r3.z ),
         vec4( r0.w, r1.w, r2.w, r3.w )
    );
    return t;
}

vec3 opTx( vec3 p, mat4 m ) {
    return (transposeM4(m)*vec4(p,1.0)).xyz;
}

vec3 rotate_x(vec3 p, float angle) {
    return opTx(p, Rot4X(angle));
}

vec3 rotate_y(vec3 p, float angle) {
    return opTx(p, Rot4Y(angle));
}

vec3 rotate_z(vec3 p, float angle) {
    return opTx(p, Rot4Z(angle));
}


float opU( float d1, float d2 ) {
    return min( d1, d2 );
}

float opS( float d1, float d2 ) {
    return max(-d1,d2);
}

float opI( float d1, float d2 ) {
    return max(d1,d2);
}


float sdBox( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}


float getSDF(int body_part_index, int x, int y, int z) {
    int i = z + y * body_part_size + x * body_part_size * body_part_size;

    float x_tex = mod(float(i), body_part_texture_size) / body_part_texture_size;
    float y_tex = floor(float(i) / body_part_texture_size) / body_part_texture_size;
    y_tex = 1. - y_tex - 1./body_part_texture_size;

    float a;

    if (body_part_index == 0) {
        a = texture2D(sdf_objects[0], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 1) {
        a = texture2D(sdf_objects[1], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 2) {
        a = texture2D(sdf_objects[2], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 3) {
        a = texture2D(sdf_objects[3], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 4) {
        a = texture2D(sdf_objects[4], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 5) {
        a = texture2D(sdf_objects[5], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 6) {
        a = texture2D(sdf_objects[6], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 7) {
        a = texture2D(sdf_objects[7], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 8) {
        a = texture2D(sdf_objects[8], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 9) {
        a = texture2D(sdf_objects[9], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 10) {
        a = texture2D(sdf_objects[10], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 11) {
        a = texture2D(sdf_objects[11], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 12) {
        a = texture2D(sdf_objects[12], vec2(x_tex, y_tex)).a;
    } else if (body_part_index == 13) {
        a = texture2D(sdf_objects[13], vec2(x_tex, y_tex)).a;
    }
    return a;
}

vec4 getTexture(int texture_index, float x, float y) {
    float numTextures = 5.;
    float x_tex = x / float(body_size);
    float y_tex = y / float(body_size);

    vec4 pixel = texture2D(image_texture, vec2((x_tex + float(texture_index)) / numTextures, y_tex));

    return pixel;
}

vec3 getSkeletonPoint(int index) {
    return vec3(
        texture2D(transformation_data, vec2(float(index*3) / 64., 0. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 1) / 64., 0. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 2) / 64., 0. / 4.)).a
    );
}

vec3 getBodyPartMidpoint(int index) {
    return vec3(
        texture2D(transformation_data, vec2(float(index*3) / 64., 1. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 1) / 64., 1. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 2) / 64., 1. / 4.)).a
    );
}

vec3 getBodyPartTranslation(int index) {
    return vec3(
        texture2D(transformation_data, vec2(float(index*3) / 64., 2. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 1) / 64., 2. / 4.)).a,
        texture2D(transformation_data, vec2(float(index*3 + 2) / 64., 2. / 4.)).a
    );
}

float getBodyPartRotation(int index, int position) {
    return texture2D(transformation_data, vec2(float(index*4 + position) / 64., 3. / 4.)).a;
}

vec3 headbangAnimation(int index, vec3 point) {
    if (index == 1) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 head_shift = getSkeletonPoint(8) - bodyPartMidpoint;

        point = translate(point, head_shift);
        point = rotate_x(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.x);
        point = rotate_z(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.z);
        point = translate(point, negate(head_shift));
    }
    return point;
}

vec3 headbangPoint(int index, vec3 point) {
    if (index == 1) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 head_shift = getSkeletonPoint(8) - bodyPartMidpoint;

        point = translate(point, bodyPartMidpoint);
        point = translate(point, head_shift);
        point = rotate_x(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.x);
        point = rotate_z(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.z);
        point = translate(point, negate(head_shift));
        point = translate(point, negate(bodyPartMidpoint));
    }

    return point;
}

vec3 headbangNormal(int index, vec3 point) {
    if (index == 1) {
        point = rotate_x(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.x);
        point = rotate_z(point, (sin(czm_frameNumber) * (PI/8.) + (PI/8.)) * headbang_rotation.z);
    }
    return point;
}

/*
vec3 walkingAnimation(int index, vec3 point) {
    if (index == 5 || index == 6 || index == 7) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 upper_leg_shift = getSkeletonPoint(2) - bodyPartMidpoint;

        point = translate(point, upper_leg_shift);
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/8.));
        point = translate(point, negate(upper_leg_shift));

        if (index == 6 || index == 7) {
            vec3 lower_leg_shift = getSkeletonPoint(1) - bodyPartMidpoint;

            point = translate(point, lower_leg_shift);
            point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/6.) - (PI/6.));
            point = translate(point, negate(lower_leg_shift));

            if (index == 7) {
                vec3 foot_shift = getSkeletonPoint(0) - bodyPartMidpoint;

                point = translate(point, foot_shift);
                point = rotate_y(point, PI / 6.);
                point = rotate_z(point, PI / 6.);
                point = translate(point, negate(foot_shift));
            }
        }
    } else if (index == 8 || index == 9 || index == 10) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 upper_leg_shift = getSkeletonPoint(3) - bodyPartMidpoint;

        point = translate(point, upper_leg_shift);
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/8.) + (PI/8.));
        point = translate(point, negate(upper_leg_shift));

        if (index == 9 || index == 10) {
            vec3 lower_leg_shift = getSkeletonPoint(4) - bodyPartMidpoint;

            point = translate(point, lower_leg_shift);
            point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/6.) + (PI/8.));
            point = rotate_y(point, PI / 4.);
            point = translate(point, negate(lower_leg_shift));
        }
    } else if (index == 2 || index == 3 || index == 4) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 right_shoulder_shift = getSkeletonPoint(12) - bodyPartMidpoint;

        point = translate(point, right_shoulder_shift);
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/4.));
        point = translate(point, negate(right_shoulder_shift));

        if (index == 3 || index == 4) {
            vec3 right_elbow_shift = getSkeletonPoint(11) - bodyPartMidpoint;

            point = translate(point, right_elbow_shift);
            point = rotate_y(point, PI/4.);
            point = translate(point, negate(right_elbow_shift));
        }
    } else if (index == 11 || index == 12 || index == 13) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 left_shoulder_shift = getSkeletonPoint(13) - bodyPartMidpoint;

        point = translate(point, left_shoulder_shift);
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/4.));
        point = translate(point, negate(left_shoulder_shift));

        if (index == 12 || index == 13) {
            vec3 left_elbow_shift = getSkeletonPoint(14) - bodyPartMidpoint;

            point = translate(point, left_elbow_shift);
            point = rotate_y(point, PI/4.);
            point = translate(point, negate(left_elbow_shift));
        }
    }

    return point;
}

vec3 walkingAnimationPoint(int index, vec3 point) {
    if (index == 5 || index == 6 || index == 7) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 upper_leg_shift = getSkeletonPoint(2) - bodyPartMidpoint;

        point = translate(point, bodyPartMidpoint);
        point = translate(point, upper_leg_shift);
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/8.));
        point = translate(point, negate(upper_leg_shift));
        point = translate(point, negate(bodyPartMidpoint));

        if (index == 6 || index == 7) {
            vec3 lower_leg_shift = getSkeletonPoint(1) - bodyPartMidpoint;

            point = translate(point, bodyPartMidpoint);
            point = translate(point, lower_leg_shift);
            point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/6.) - (PI/6.));
            point = translate(point, negate(lower_leg_shift));
            point = translate(point, negate(bodyPartMidpoint));

            if (index == 7) {
                vec3 foot_shift = getSkeletonPoint(0) - bodyPartMidpoint;

                point = translate(point, bodyPartMidpoint);
                point = translate(point, foot_shift);
                point = rotate_y(point, PI / 6.);
                point = rotate_z(point, PI / 6.);
                point = translate(point, negate(foot_shift));
                point = translate(point, negate(bodyPartMidpoint));
            }
        }
    } else if (index == 8 || index == 9 || index == 10) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 upper_leg_shift = getSkeletonPoint(3) - bodyPartMidpoint;

        point = translate(point, bodyPartMidpoint);
        point = translate(point, upper_leg_shift);
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/8.) + (PI/8.));
        point = translate(point, negate(upper_leg_shift));
        point = translate(point, negate(bodyPartMidpoint));

        if (index == 9 || index == 10) {
            vec3 lower_leg_shift = getSkeletonPoint(4) - bodyPartMidpoint;

            point = translate(point, bodyPartMidpoint);
            point = translate(point, lower_leg_shift);
            point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/6.) + (PI/8.));
            point = rotate_y(point, PI / 4.);
            point = translate(point, negate(lower_leg_shift));
            point = translate(point, negate(bodyPartMidpoint));
        }
    } else if (index == 2 || index == 3 || index == 4) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 right_shoulder_shift = getSkeletonPoint(12) - bodyPartMidpoint;

        point = translate(point, bodyPartMidpoint);
        point = translate(point, right_shoulder_shift);
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/4.));
        point = translate(point, negate(right_shoulder_shift));
        point = translate(point, negate(bodyPartMidpoint));

        if (index == 3 || index == 4) {
            vec3 right_elbow_shift = getSkeletonPoint(11) - bodyPartMidpoint;

            point = translate(point, bodyPartMidpoint);
            point = translate(point, right_elbow_shift);
            point = rotate_y(point, PI/4.);
            point = translate(point, negate(right_elbow_shift));
            point = translate(point, negate(bodyPartMidpoint));
        }
    } else if (index == 11 || index == 12 || index == 13) {
        vec3 bodyPartMidpoint = getBodyPartMidpoint(index);
        vec3 left_shoulder_shift = getSkeletonPoint(13) - bodyPartMidpoint;

        point = translate(point, bodyPartMidpoint);
        point = translate(point, left_shoulder_shift);
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/4.));
        point = translate(point, negate(left_shoulder_shift));
        point = translate(point, negate(bodyPartMidpoint));

        if (index == 12 || index == 13) {
            vec3 left_elbow_shift = getSkeletonPoint(14) - bodyPartMidpoint;

            point = translate(point, bodyPartMidpoint);
            point = translate(point, left_elbow_shift);
            point = rotate_y(point, PI/4.);
            point = translate(point, negate(left_elbow_shift));
            point = translate(point, negate(bodyPartMidpoint));
        }
    }

    return point;
}

vec3 walkingAnimationNormal(int index, vec3 point) {
    if (index == 5 || index == 6 || index == 7) {
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/8.));

        if (index == 6 || index == 7) {
            point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/6.) - (PI/6.));

            if (index == 7) {
                point = rotate_y(point, PI / 6.);
                point = rotate_z(point, PI / 6.);
            }
        }
    } else if (index == 8 || index == 9 || index == 10) {
        point = rotate_y(point, -PI / 4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/8.) + (PI/8.));

        if (index == 9 || index == 10) {
            point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/6.) + (PI/8.));
            point = rotate_y(point, PI / 4.);
        }
    } else if (index == 2 || index == 3 || index == 4) {
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3 + PI) * (PI/4.));

        if (index == 3 || index == 4) {
            point = rotate_y(point, PI/4.);
        }
    } else if (index == 11 || index == 12 || index == 13) {
        point = rotate_y(point, -PI/4.);
        point = rotate_z(point, sin(czm_frameNumber * 0.3) * (PI/4.));

        if (index == 12 || index == 13) {
            point = rotate_y(point, PI/4.);
        }
    }

    return point;
}
*/



obj2 shapes(vec3 inputPoint) {
    vec3 transformedPosition = position.yzx;
    inputPoint = translate(inputPoint, transformedPosition * MINIFIER);
    inputPoint = rotate_x(inputPoint, model_rotation.x);
    inputPoint = rotate_y(inputPoint, model_rotation.y);
    inputPoint = rotate_z(inputPoint, model_rotation.z);

    float x = inputPoint[0], y = inputPoint[1], z = inputPoint[2];
    float distance = 10000.;
    float min_distance = 10000.;
    int min_index;
    float distance2;

    for (int i = 0; i < 14; i++) {
        vec3 object_translation = getBodyPartMidpoint(i);
        vec3 point = translate(vec3(x, y, z), object_translation);

        point = headbangAnimation(i, point);
        //point = walkingAnimation(i, point);

        /* 3.6M animation*/
        vec3 body_part_translation = getBodyPartTranslation(i);
        point = translate(point, body_part_translation);

        point = rotate_y(point, getBodyPartRotation(i, 3));
        point = rotate_z(point, -getBodyPartRotation(i, 2));
        point = rotate_z(point, getBodyPartRotation(i, 1));
        point = rotate_y(point, -getBodyPartRotation(i, 0));


        /*
        //arm waving
        if (i == 3 || i == 4) {
            vec3 skeleton_point = getSkeletonPoint(11);
            vec3 elbow_shift = skeleton_point - object_translation;
            point = translate(point, elbow_shift);
            point = rotate_z(point, sin(czm_frameNumber * 0.1) * (PI/4.) + PI/8.);
            point = translate(point, negate(elbow_shift));
        }
        */

        float scale = scales[i];
        float scale_half = 0.5 * scale ;
        float bounding_cube_size = (body_part_size_half - 2.) * scale / float(body_part_size);

        float gx = point.x / scale_half * body_part_size_half + body_part_size_half;
        float gy = point.y / scale_half * body_part_size_half + body_part_size_half;
        float gz = point.z / scale_half * body_part_size_half + body_part_size_half;
        int x1 = int(gx);
        int y1 = int(gy);
        int z1 = int(gz);

        if (x1 >= 0 && x1 < body_part_size - 1 && y1 >= 0 && y1 < body_part_size - 1 && z1 >= 0 && z1 < body_part_size - 1) {
            /*distance2 = getSDF(i, x1, y1, z1) / body_size_half * scale * 2.;*/

            //trilinear interpolation
            float dx = gx - float(x1);
            float dy = gy - float(y1);
            float dz = gz - float(z1);

            float c00 = getSDF(i, x1, y1, z1) * (1. - dx) + getSDF(i, x1+1, y1, z1) * dx;
            float c01 = getSDF(i, x1, y1, z1+1) * (1. - dx) + getSDF(i, x1+1, y1, z1+1) * dx;
            float c10 = getSDF(i, x1, y1+1, z1) * (1. - dx) + getSDF(i, x1+1, y1+1, z1) * dx;
            float c11 = getSDF(i, x1, y1+1, z1+1) * (1. - dx) + getSDF(i, x1+1, y1+1, z1+1) * dx;

            float c0 = c00 * (1. - dy) + c10 * dy;
            float c1 = c01 * (1. - dy) + c11 * dy;

            distance2 = c0 * (1. - dz) + c1 * dz;
            distance2 = distance2 / body_size_half * scale * 2.;

        } else {
            distance2 = sdBox(point, vec3(bounding_cube_size, bounding_cube_size, bounding_cube_size));
        }

        distance = min(distance, distance2);
        min_distance = min(min_distance, distance2);

        if (distance == distance2) {
            min_index = i;
        }
    }

    obj2 a;
    a.min_index = min_index;
    a.distance = distance;

    return a;
}

vec3 calcNormal(vec3 intersectionPoint) {
    vec3 eps = vec3( 0.1, 0.0, 0.0 );
    vec3 nor = vec3(
        shapes(intersectionPoint+eps.xyy).distance - shapes(intersectionPoint-eps.xyy).distance,
        shapes(intersectionPoint+eps.yxy).distance - shapes(intersectionPoint-eps.yxy).distance,
        shapes(intersectionPoint+eps.yyx).distance - shapes(intersectionPoint-eps.yyx).distance
    );
    return normalize(nor);
}

vec4 getRGBColor(int index) {
    if (index == 0) {
        return vec4(0.25, 0.25, 0.25, 1.);
    }
    if (index == 1) {
        return vec4(0.75, 0.75, 0.75, 1.);
    }
    if (index == 2) {
        return vec4(1.0, 0.0, 0.0, 1.);
    }
    if (index == 3) {
        return vec4(0.0, 1.0, 0.0, 1.);
    }
    if (index == 4) {
        return vec4(0.0, 0.0, 1.0, 1.);
    }
    if (index == 5) {
        return vec4(1.0, 1.0, 0.0, 1.);
    }
    if (index == 6) {
        return vec4(0.0, 1.0, 1.0, 1.);
    }
    if (index == 7) {
        return vec4(1.0, 0.0, 1.0, 1.);
    }
    if (index == 8) {
        return vec4(0.5, 0.5, 0.0, 1.);
    }
    if (index == 9) {
        return vec4(0.0, 0.5, 0.5, 1.);
    }
    if (index == 10) {
        return vec4(0.5, 0.0, 0.5, 1.);
    }
    if (index == 11) {
        return vec4(0.5, 0.0, 0.0, 1.);
    }
    if (index == 12) {
        return vec4(0.0, 0.5, 0.0, 1.);
    }
    if (index == 13) {
        return vec4(0.0, 0.0, 0.5, 1.);
    }
}

float getDistinctAngle(int index) {
    if (index == 0) {
        return 0.;
    } else if(index == 1) {
        return PI / 2.;
    } else if(index == 2) {
        return PI;
    } else if(index == 3) {
        return 3. * PI / 2.;
    }
}

float getFactor(int index, float angle) {
    if (index == 0) {
        return 2. * abs(angle/PI - 1.) - 1.;
    } else if(index == 1) {
        return -2. * abs(angle/PI - 0.5) + 1.;
    } else if(index == 2) {
        return -2. * abs(angle/PI - 1.) + 1.;
    } else if(index == 3) {
        return -2. * abs(angle/PI - 1.5) + 1.;
    }
}


void main() {
    float u = gl_FragCoord.x * 2.0 / czm_viewport.z - 1.0;
    float v = gl_FragCoord.y * 2.0 / czm_viewport.w - 1.0;

    vec3 rayDirection = normalize(cameraForward + cameraRight * u * tanCameraFovHalf + cameraUp * v * tanCameraFovYHalf);

    rayDirection = rayDirection.yzx;

    float maxDistance = czm_currentFrustum.y * 2. * MINIFIER;
    float distance;
    float t = 0.0;
    bool hit = false;

    vec3 positionWC = czm_viewerPositionWC * MINIFIER;
    positionWC = positionWC.yzx;

    for (float i = 0.; i < 1000.; i++) {
        vec3 p = positionWC + rayDirection * t;

        obj2 result = shapes(p);
        distance = result.distance;

        if (distance < EPSILON) {
            int min_index = result.min_index;
            //vec4 pixel_color = getRGBColor(min_index);

            vec3 normal = calcNormal(p);

            normal = rotate_x(normal, model_rotation.x);
            normal = rotate_y(normal, model_rotation.y);
            normal = rotate_z(normal, model_rotation.z);

            vec3 transformedPosition = position.yzx;
            p = translate(p, transformedPosition * MINIFIER);
            p = rotate_x(p, model_rotation.x);
            p = rotate_y(p, model_rotation.y);
            p = rotate_z(p, model_rotation.z);

            normal = headbangNormal(min_index, normal);
            p = headbangPoint(min_index, p);

            //normal = walkingAnimationNormal(min_index, normal);
            //p = walkingAnimationPoint(min_index, p);

            /*
            if (min_index == 3 || min_index == 4) {
                vec3 object_translation = getBodyPartTranslation(min_index);
                vec3 skeleton_point = getSkeletonPoint(11);
                vec3 elbow_shift = skeleton_point - object_translation;

                p = translate(p, object_translation);
                p = translate(p, elbow_shift);
                p = rotate_z(p, sin(czm_frameNumber * 0.1) * (PI/4.) + PI/8.);
                p = translate(p, negate(elbow_shift));
                p = translate(p, negate(object_translation));

                normal = rotate_z(normal, sin(czm_frameNumber * 0.1) * (PI/4.) + PI/8.);
            }
            */

            /* 3.6M animation*/
            vec3 body_part_translation = getBodyPartTranslation(min_index);
            p = translate(p, body_part_translation);

            p = rotate_y(p, getBodyPartRotation(min_index, 3));
            p = rotate_z(p, -getBodyPartRotation(min_index, 2));
            p = rotate_z(p, getBodyPartRotation(min_index, 1));
            p = rotate_y(p, -getBodyPartRotation(min_index, 0));

            normal = rotate_y(normal, getBodyPartRotation(min_index, 3));
            normal = rotate_z(normal, -getBodyPartRotation(min_index, 2));
            normal = rotate_z(normal, getBodyPartRotation(min_index, 1));
            normal = rotate_y(normal, -getBodyPartRotation(min_index, 0));


            //float lightIntensity = abs(dot(normal, rayDirection));

            //gl_FragColor = vec4(pixel_color.rgb  * lightIntensity, 1.);

            //gl_FragColor = getTexture(0, p.x + body_size_half, p.y + body_size_half);
            //gl_FragColor = vec4(normal.xyz, 1.);

            float angle = atan(normal.x, normal.z);
            angle = mod(angle, 2.*PI);
            /*
                int texture_index = int(floor(mod(angle / (2.*PI) * 4., 3.5) + 0.5));
                angle = (float(texture_index) / 4.) * 2.*PI;

                float u = p.x * cos(angle) - sin(angle) * p.z;
                float v = p.y;
                vec4 texture_pixel = getTexture(texture_index, u + body_size_half, v + body_size_half);
            */


            vec3 pixel_color = vec3(0., 0., 0.);

            for (int texture_index = 0; texture_index < 4; texture_index++) {
                float texture_factor = getFactor(texture_index, angle);

                if (texture_factor < 0.)
                    continue;

                float distinct_angle = getDistinctAngle(texture_index);

                //texture blending
                float u = p.x * cos(distinct_angle) - sin(distinct_angle) * p.z;
                float v = p.y;
                vec4 texture_pixel = getTexture(texture_index, u + body_size_half, v + body_size_half);

                //use generated front texture if original one does not provide a value
                if (texture_pixel.a == 0.)
                    texture_pixel = getTexture(4, u + body_size_half, v + body_size_half);

                pixel_color += texture_pixel.rgb * texture_factor;
            }

            gl_FragColor = vec4(pixel_color.rgb, 1.);

            float distanceWC = t * (1. / MINIFIER);
            float cosineAngle = dot(cameraForward.yzx, rayDirection);
            float z = cosineAngle * distanceWC;
            float depth = depthA - depthB/z;
            float fragDepth = (1. + depth) * 0.5;

            gl_FragDepthEXT = fragDepth;

            hit = true;
            break;
        }

        t += distance;

        if (t > maxDistance) {
            break;
        }
    }

    //gl_FragColor = vec4((rayDirection.x + 1.)/2., (rayDirection.y + 1.)/2., (rayDirection.z + 1.)/2., 1.);
    //gl_FragColor = vec4(float(j) / 1000., float(j) / 1000., float(j) / 1000., 1.);

    if (!hit) {
        discard;
        //gl_FragColor = vec4(1., 1., 1., .1);
    }
}


void main3() {
    int x = int(gl_FragCoord.x / czm_viewport.z * 64.);
    int y = int(gl_FragCoord.y / czm_viewport.w * 64.);

    float minValue = 0.;

    for (int z = 0; z < 64; z++) {
        float a = getSDF(0, y, z, x);
        /*
        int i = z + y * 64 + x * 64 * 64;
        float x_screen = mod(float(i), 512.) / 512.;
        float y_screen = floor(float(i) / 512.) / 512.;

        float a = texture2D(sdf_objects[0], vec2(x_screen, 1. - y_screen - 1./512.)).a;
        */
        minValue = min(minValue, a);
    }

    if (minValue == 0.)
        discard;

    gl_FragColor = vec4(1., 1., 1., 1.);
}

void main2() {
    float x = gl_FragCoord.x / czm_viewport.z * float(body_size);
    float y = gl_FragCoord.y / czm_viewport.w * float(body_size);

    vec4 rgba = getTexture(1, x, y);

    gl_FragColor = rgba;
}
