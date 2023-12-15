#extension GL_EXT_frag_depth : enable

uniform vec3 cameraForward;
uniform vec3 cameraUp;
uniform vec3 cameraRight;
uniform float tanCameraFovHalf;
uniform float tanCameraFovYHalf;
uniform float depthA;
uniform float depthB;

const float MINIFIER = 0.000001;
const float EPSILON = 0.001;
const int MAX_STEPS = 64;
const float PI = 3.141592654;
const float SQRT2 = 1.41421356;


float raySphere(vec3 p, float r) {
    return length(p) - r;
}

float rayHemisphere( vec3 p, float r) {
    return max(length(p) - r, p.x);
}

float rayBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return max(d.x, max(d.y, d.z));
}

float rayCylinder( vec3 p, float radius, float height ) {
    return max(length(p.xy) - radius, abs(p.z) - height);
}


//t.x - inner (empty) radius, t.y - radius of torus
float rayTorus( vec3 p, vec2 t ) {
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float coneTest(vec3 p, float radius, float height) {
    if (p.z > height)
        return length(vec3(0, 0, height) - p);

    if (p.z < 0.) {
        float cylinderDistance = min(length(p.xy) - radius, radius);
        return sqrt(cylinderDistance*cylinderDistance + p.z*p.z);
    }


    float horizontalDistance = length(p.xy) - radius*(height - abs(p.z));
    float capAngle = atan(radius/height);

    return horizontalDistance / cos(capAngle);
}

float rayOctaeder(vec3 p) {
    //if you change the factor of the coordinates, you have to adjust the root (a² + b² + c², by default 3 if a = b = c = 1)
    return (abs(p.x * 2.) + abs(p.y) + abs(p.z) - 1.) / sqrt(6.);
}

float rayPyramid(vec3 p, float bottomSquareLength, float topSquareLength, float height) {
    float factor_x = bottomSquareLength;
    float factor_y = bottomSquareLength;
    float factor_z = 1. / (height * (1. / (1. - (bottomSquareLength - topSquareLength))));

    //return max((abs(p.x) + abs(p.z) - 1.) / 1.41421356, max((abs(p.x) + abs(p.y) - 1.) / 1.41421356, abs(p.z) - 1.));
    return max((factor_x*abs(p.x) + factor_y*abs(p.y) + factor_z*abs(p.z) - 1.) / sqrt(factor_x*factor_x + factor_y*factor_y + factor_z*factor_z), -p.z);
}

float rayPyramidFrustum(vec3 p, float bottomSquareLength, float topSquareLength, float height) {
    float z_intersection_plane;

    if (bottomSquareLength > topSquareLength) {
        p.z = p.z + height * 0.5;
        z_intersection_plane = abs(p.z - height*0.5) - height*0.5;

    } else {
        float tmp = bottomSquareLength;
        bottomSquareLength = topSquareLength;
        topSquareLength = tmp;

        p.z = p.z - height * 0.5;
        z_intersection_plane = abs(p.z + height*0.5) - height*0.5;
    }

    float factor_x = 1. / bottomSquareLength;
    float factor_y = 1. / bottomSquareLength;
    float factor_z = (1. - (topSquareLength / bottomSquareLength)) / height;

    return max((factor_x*abs(p.x) + factor_y*abs(p.y) + factor_z*abs(p.z) - 1.) / sqrt(factor_x*factor_x + factor_y*factor_y + factor_z*factor_z), z_intersection_plane);
}



float rayTriangle(vec3 p) {
    float square = length(min(p.xy, 0.));
    float triangle = (max(p.x, 0.) + max(p.y, 0.) - 1.) / 1.41421356;
    return max(max(square, triangle), abs(p.z) - 1.);
}

float testTriangle2(vec3 p) {
    float square = max(-p.x, 0.);
    float triangle = (abs(p.x) + abs(p.y) - 1.) / 1.41421356;
    return max(max(square, triangle), abs(p.z) - 1.);
}

float rayPieSegment(vec3 q, float startAngle, float endAngle, float height, float explosionTranslation) {
    float x_start = cos(startAngle);
    float y_start = sin(startAngle);
    float x_middle = cos(startAngle + (endAngle - startAngle) * 0.5);
    float y_middle = sin(startAngle + (endAngle - startAngle) * 0.5);
    float x_end = cos(endAngle);
    float y_end = sin(endAngle);

    vec3 p = q - vec3(x_middle * explosionTranslation, -y_middle * explosionTranslation, 0.);

    float circle = length(p.xy) - 1.;
    float triangleStart = (x_start*p.y + y_start*p.x);
    float triangleEnd = (-y_end*p.x - x_end*p.y);
    float triangle;

    if (endAngle - startAngle < PI)
        triangle = max(triangleStart, triangleEnd);
    else
        triangle = min(triangleStart, triangleEnd);

    return max(max(triangle, circle), abs(p.z - height * 0.5) - height * 0.5);
}



float testBox2( vec3 p, vec3 b ) {
    return length(max(abs(p) - b, 0.));
}

float testPieSegment2( vec3 p ) {
    //float angle = abs(atan(p.y/p.x));
    return max(max(0.5 - normalize(p.xy).x, length(p.xy) - 1.), abs(p.z) - 0.5);
}

struct obj {
    float distance;
    vec4 color;
};


obj rayUnion(obj o1, obj o2) {
    if (o1.distance < o2.distance)
        return o1;
    else
        return o2;
}

obj raySubtraction( obj o1, obj o2 ) {
    if (-o1.distance > o2.distance)
        return o1;
    else
        return o2;
}

obj rayIntersection(obj o1, obj o2) {
    if (o1.distance > o2.distance)
        return o1;
    else
        return o2;
}

vec3 rayTranslation(vec3 p, vec3 d) {
    return p - d;
}

vec3 rayTranslationZ(vec3 p, float dz) {
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

vec3 rayRotationX(vec3 p, float angle) {
    return opTx(p, Rot4X(angle));
}

vec3 rayRotationY(vec3 p, float angle) {
    return opTx(p, Rot4Y(angle));
}

vec3 rayRotationZ(vec3 p, float angle) {
    return opTx(p, Rot4Z(angle));
}

vec3 opTwist( vec3 p ) {
    float new_x = p.x + cos((p.z - 1.)*PI);
    float new_y = p.y + sin((p.z - 1.)*PI);

    vec3 rot = opTx( vec3(p.x / length(p.xy), p.y / length(p.xy), 0.), Rot4X( PI/2.0 ) );

    //vec3 rotz = opTx( vec3(rotx.x, rotx.y, rotx.z), Rot4Z(PI * (p.z - 1.)));
    return vec3(p.x, p.y, p.z);
    //return vec3(new_x * 0.25, new_y * 0.25, p.z + rot.z);
}


float rayTorusSegment( vec3 p, vec2 t, float startAngle, float endAngle ) {
    //float new_pz = p.z + atan(p.y/p.x); helix test

    float x_start = cos(startAngle);
    float y_start = sin(startAngle);
    float x_end = cos(endAngle);
    float y_end = sin(endAngle);

    float triangleStart = (x_start*p.y + y_start*p.x);
    float triangleEnd = (-y_end*p.x - x_end*p.y);
    float triangle;

    if (endAngle - startAngle < PI)
        triangle = max(triangleStart, triangleEnd);
    else
        triangle = min(triangleStart, triangleEnd);


    vec2 q = vec2(length(p.xy)-t.x, p.z);
    float torus = length(q)-t.y;
    return max(triangle, torus);//max(triangle, torus);
}

float opCheapBend( vec3 p )
{
    float c = cos(0.1*p.y);
    float s = sin(0.1*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.yx,p.z);
    return rayBox(q, vec3(2., 1., 0.5));
}

//heights, lengths, colors
obj createStackedCuboid(vec3 p, float h1, float h2, float h3, float l1, float l2, float l3, vec3 c1, vec3 c2, vec3 c3) {
    float previousHeight = h1 * 0.5;
    vec3 p1 = rayTranslation(p, vec3(0., 0., previousHeight));

    obj b1;
    b1.distance = rayBox(p1, vec3(l1, l1, h1));
    b1.color = vec4(c1, 1.);

    previousHeight += h1 + h2;
    vec3 p2 = rayTranslation(p, vec3((l1 - l2), (l1 - l2), previousHeight));

    obj b2;
    b2.distance = rayBox(p2, vec3(l2, l2, h2));
    b2.color = vec4(c2, 1.);

    previousHeight += h2 + h3;
    vec3 p3 = rayTranslation(p, vec3((l1 - l3), (l1 - l3), previousHeight));

    obj b3;
    b3.distance = rayBox(p3, vec3(l3, l3, h3));
    b3.color = vec4(c3, 1.);

    return rayUnion(rayUnion(b1, b2), b3);
}

// source: https://gamedev.stackexchange.com/questions/96459/fast-ray-sphere-collision-code
bool intersectsBoundingSphere(vec3 rayOrigin, vec3 rayDirection, vec3 spherePosition, float sphereRadius) {
    vec3 p = rayOrigin - spherePosition;

    float rSquared = sphereRadius * sphereRadius;
    float p_d = dot(p, rayDirection);

    if (p_d > 0. || dot(p, p) < rSquared)
        return false;

    vec3 a = p - p_d * rayDirection;

    float aSquared = dot(a, a);

    if (aSquared > rSquared)
        return false;

    return true;
}

obj mapStackedCuboids(vec3 p) { //
    float h1 = 0.05;
    float h2 = 0.025;
    float h3 = 0.01;

    float l1 = 0.1;
    float l2 = 0.05;
    float l3 = 0.025;

    vec3 c1 = vec3(3.,78.,123.)/255.;
    vec3 c2 = vec3(54.,144.,192.)/255.;
    vec3 c3 = vec3(204.,235.,255.)/255.;

    obj stackedCuboids = createStackedCuboid(p, h1, h2, h3, l1, l2, l3, c1, c2, c3);

    for (float i = -2.; i < 3.; i++) {
        for (float j = -2.; j < 3.; j++) {
            /*vec3 q = rayRotationY(p, i * PI / 30.);
            q = rayRotationZ(q, j * PI / 30.); */
            vec3 q = rayTranslation(p, vec3(7., i * 0.5, j * 0.5));
            q = rayRotationY(q, - PI / 2.);

            obj stackedCuboid = createStackedCuboid(q, h1, h2, h3, l1, l2, l3, c1, c2, c3);
            stackedCuboids = rayUnion(stackedCuboid, stackedCuboids);

            /*obj boundingSphere;
            boundingSphere.distance = raySphere(rayTranslation(q, vec3(0., 0., 0.05)), 0.17);
            boundingSphere.color = vec4(1., 1., 1., 1.);

            stackedCuboids = rayUnion(stackedCuboids, boundingSphere); */
        }
    }

    return stackedCuboids;
}

bool checkIfRayIntersectsBoundingSpheres(vec3 rayOrigin, vec3 rayDirection) {
    for (float i = -3.; i < 4.; i++) {
        for (float j = -3.; j < 4.; j++) {
            bool intersects = intersectsBoundingSphere(rayOrigin, rayDirection, vec3(7., i * 0.5, j * 0.5 + 0.05), 0.17);

            if (intersects)
                return true;
        }
    }

    return false;
}

obj mapOneStackedCuboid(vec3 p) {
    p = rayTranslation(p, vec3(7., 0., 0.));
    p = rayRotationY(p, -PI / 2.);

    float h1 = 0.05;
    float h2 = 0.025;
    float h3 = 0.01;

    float l1 = 0.1;
    float l2 = 0.05;
    float l3 = 0.025;

    vec3 c1 = vec3(3.,78.,123.)/255.;
    vec3 c2 = vec3(54.,144.,192.)/255.;
    vec3 c3 = vec3(204.,235.,255.)/255.;

    return createStackedCuboid(p, h1, h2, h3, l1, l2, l3, c1, c2, c3);
}

obj mapRedBox(vec3 p) {
    obj b;
    b.distance = rayBox(p, vec3(1., 0.5, 0.25));
    b.color = vec4(1., 0., 0., 1.);

    return b;
}


obj mapExtrudeStackedCuboid(vec3 p) {
    float between0and1 = cos(czm_frameNumber * 0.01) * 0.5 + 0.5;
    p = rayTranslation(p, vec3(0., 5., 5.));
    //p = opTx( p, Rot4X( PI/4.0 ) );

    float h1 = 0.5;
    float h2 = 0.25;
    float h3 = 0.1;

    float previousHeight = h1 * between0and1 * 0.5;
    vec3 p1 = rayTranslation(p, vec3(0., 0., previousHeight));

    obj b1;
    b1.distance = rayBox(p1, vec3(1., 1., h1 * between0and1));
    b1.color = vec4(vec3(3.,78.,123.)/255., 1.);

    previousHeight += h1 * between0and1 + h2 * between0and1;
    vec3 p2 = rayTranslation(p, vec3((1. - 0.5), (1. - 0.5), previousHeight));

    obj b2;
    b2.distance = rayBox(p2, vec3(0.5, 0.5, h2 * between0and1));
    b2.color = vec4(vec3(54.,144.,192.)/255., 1.);

    previousHeight += h2 * between0and1 + h3 * between0and1;
    vec3 p3 = rayTranslation(p, vec3((1. - 0.25), (1. - 0.25), previousHeight));

    obj b3;
    b3.distance = rayBox(p3, vec3(0.25, 0.25, h3 * between0and1));
    b3.color = vec4(vec3(204.,235.,255.)/255., 1.);

    return rayUnion(rayUnion(b1, b2), b3);
}

obj createMorphedHemisphere(vec3 p, float radiusMonth, vec4 colorMonth, float angleMonth, vec3 offsetMonth, float radiusSeason, vec4 colorSeason, float angleSeason, vec3 offsetSeason, float between0and1, vec3 periodicTranslation) {
    vec3 pMonth = rayRotationZ(p, angleMonth);
    pMonth = rayTranslation(rayTranslation(pMonth, vec3(-7.5, 0., 0.)), offsetMonth);

    vec3 pSeason = rayRotationZ(p, angleSeason);
    pSeason = rayTranslation(rayTranslation(pSeason, vec3(-7.5, 0., 0.)), offsetSeason);

    obj m1;
    m1.distance = rayHemisphere(rayTranslation(pMonth, periodicTranslation), radiusMonth) * between0and1 + rayHemisphere(rayTranslation(pSeason, periodicTranslation), radiusSeason) * (1. - between0and1);
    m1.color = colorMonth * between0and1 + colorSeason * (1. - between0and1);

    return m1;
}


//http://www.intmath.com/functions-and-graphs/graphs-using-svg.php
float between0and1 = min(max(2. - abs(mod(czm_frameNumber * 0.05, 6.) - 3.), 0.), 1.);
float between0and1Sin = min(max(1.5 - abs(mod(czm_frameNumber * 0.05, 3.) - 1.5), 0.0001), 1.);

obj mapMorphedHemispheres(vec3 p) { //MorphedHemispheres
    p = rayTranslation(p, vec3(0., 7., 7.));
    p = opTx( p, Rot4X( PI/4.0 ) );
    p = opTx( p, Rot4Z( PI/2.0 ) );


obj son_2140_sep_2140 = createMorphedHemisphere(p, 0.625375, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 0.1, vec3(0.3126875, 0., -0.0), 1.347, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 0.623598775598, vec3(0.6735, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj son_2140_okt_0120 = createMorphedHemisphere(p, 0.492975, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 0.623598775598, vec3(0.2464875, 0., -0.0), 1.347, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 0.623598775598, vec3(0.6735, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj son_2140_nov_2140 = createMorphedHemisphere(p, 0.28465, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 1.1471975512, vec3(0.142325, 0., -0.0), 1.347, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 0.623598775598, vec3(0.6735, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj son_0 = rayUnion(rayUnion(son_2140_sep_2140, son_2140_okt_0120), son_2140_nov_2140);
obj son_0120_sep_4160 = createMorphedHemisphere(p, 0.57745, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 0.1, vec3(0.3126875, 0., -0.047925), 1.31325, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 0.623598775598, vec3(0.6735, 0., -0.03375), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj son_0120_okt_2140 = createMorphedHemisphere(p, 0.436975, vec4(vec3(1.0, 0.9019607843137255, 0.6901960784313725), 1.0), 0.623598775598, vec3(0.2464875, 0., -0.056), 1.31325, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 0.623598775598, vec3(0.6735, 0., -0.03375), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj son_0120_nov_0120 = createMorphedHemisphere(p, 0.243, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 1.1471975512, vec3(0.142325, 0., -0.04165), 1.31325, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 0.623598775598, vec3(0.6735, 0., -0.03375), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj son_1 = rayUnion(rayUnion(son_0120_sep_4160, son_0120_okt_2140), son_0120_nov_0120);
obj son_4160_sep_0120 = createMorphedHemisphere(p, 0.577275, vec4(vec3(1.0, 1.0, 0.8666666666666667), 1.0), 0.1, vec3(0.3126875, 0., -0.0481), 1.1651, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 0.623598775598, vec3(0.6735, 0., -0.1819), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj son_4160_okt_8100 = createMorphedHemisphere(p, 0.372425, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 0.623598775598, vec3(0.2464875, 0., -0.12055), 1.1651, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 0.623598775598, vec3(0.6735, 0., -0.1819), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj son_4160_nov_4160 = createMorphedHemisphere(p, 0.23745, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 1.1471975512, vec3(0.142325, 0., -0.0472), 1.1651, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 0.623598775598, vec3(0.6735, 0., -0.1819), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj son_2 = rayUnion(rayUnion(son_4160_sep_0120, son_4160_okt_8100), son_4160_nov_4160);
obj son_8100_sep_8100 = createMorphedHemisphere(p, 0.416425, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 0.1, vec3(0.3126875, 0., -0.20895), 0.98585, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 0.623598775598, vec3(0.6735, 0., -0.36115), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj son_8100_okt_4160 = createMorphedHemisphere(p, 0.3502, vec4(vec3(1.0, 0.8, 0.5137254901960784), 1.0), 0.623598775598, vec3(0.2464875, 0., -0.142775), 0.98585, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 0.623598775598, vec3(0.6735, 0., -0.36115), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj son_8100_nov_8100 = createMorphedHemisphere(p, 0.197, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 1.1471975512, vec3(0.142325, 0., -0.08765), 0.98585, vec4(vec3(0.9607843137254902, 0.5725490196078431, 0.20784313725490197), 1.0), 0.623598775598, vec3(0.6735, 0., -0.36115), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj son_3 = rayUnion(rayUnion(son_8100_sep_8100, son_8100_okt_4160), son_8100_nov_8100);
obj son_6180_sep_6180 = createMorphedHemisphere(p, 0.3866, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 0.1, vec3(0.3126875, 0., -0.238775), 0.87425, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 0.623598775598, vec3(0.6735, 0., -0.47275), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj son_6180_okt_6180 = createMorphedHemisphere(p, 0.303975, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 0.623598775598, vec3(0.2464875, 0., -0.189), 0.87425, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 0.623598775598, vec3(0.6735, 0., -0.47275), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj son_6180_nov_6180 = createMorphedHemisphere(p, 0.183675, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 1.1471975512, vec3(0.142325, 0., -0.100975), 0.87425, vec4(vec3(0.996078431372549, 0.7019607843137254, 0.33725490196078434), 1.0), 0.623598775598, vec3(0.6735, 0., -0.47275), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj son_4 = rayUnion(rayUnion(son_6180_sep_6180, son_6180_okt_6180), son_6180_nov_6180);
obj mam_4160_mar_2140 = createMorphedHemisphere(p, 0.111975, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 1.67079632679, vec3(0.0559875, 0., -0.0), 1.206775, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj mam_4160_apr_4160 = createMorphedHemisphere(p, 0.2994, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 2.19439510239, vec3(0.1497, 0., -0.0), 1.206775, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj mam_4160_mai_0120 = createMorphedHemisphere(p, 0.845325, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 2.71799387799, vec3(0.4226625, 0., -0.0), 1.206775, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj mam_0 = rayUnion(rayUnion(mam_4160_mar_2140, mam_4160_apr_4160), mam_4160_mai_0120);
obj mam_0120_mar_4160 = createMorphedHemisphere(p, 0.110225, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 1.67079632679, vec3(0.0559875, 0., -0.00175), 1.164825, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.04195), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj mam_0120_apr_2140 = createMorphedHemisphere(p, 0.239825, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 2.19439510239, vec3(0.1497, 0., -0.059575), 1.164825, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.04195), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj mam_0120_mai_4160 = createMorphedHemisphere(p, 0.79715, vec4(vec3(0.4549019607843137, 0.7686274509803922, 0.4627450980392157), 1.0), 2.71799387799, vec3(0.4226625, 0., -0.048175), 1.164825, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.04195), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj mam_1 = rayUnion(rayUnion(mam_0120_mar_4160, mam_0120_apr_2140), mam_0120_mai_4160);
obj mam_2140_mar_0120 = createMorphedHemisphere(p, 0.10795, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 1.67079632679, vec3(0.0559875, 0., -0.004025), 1.12995, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.076825), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj mam_2140_apr_8100 = createMorphedHemisphere(p, 0.2165, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 2.19439510239, vec3(0.1497, 0., -0.0829), 1.12995, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.076825), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj mam_2140_mai_2140 = createMorphedHemisphere(p, 0.77815, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 2.71799387799, vec3(0.4226625, 0., -0.067175), 1.12995, vec4(vec3(0.7294117647058823, 0.8941176470588236, 0.7019607843137254), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.076825), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj mam_2 = rayUnion(rayUnion(mam_2140_mar_0120, mam_2140_apr_8100), mam_2140_mai_2140);
obj mam_8100_mar_8100 = createMorphedHemisphere(p, 0.106225, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 1.67079632679, vec3(0.0559875, 0., -0.00575), 0.933125, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.27365), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj mam_8100_apr_0120 = createMorphedHemisphere(p, 0.21155, vec4(vec3(0.9294117647058824, 0.9725490196078431, 0.9137254901960784), 1.0), 2.19439510239, vec3(0.1497, 0., -0.08785), 0.933125, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.27365), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj mam_8100_mai_8100 = createMorphedHemisphere(p, 0.6104, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 2.71799387799, vec3(0.4226625, 0., -0.234925), 0.933125, vec4(vec3(0.09803921568627451, 0.5333333333333333, 0.25098039215686274), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.27365), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj mam_3 = rayUnion(rayUnion(mam_8100_mar_8100, mam_8100_apr_0120), mam_8100_mai_8100);
obj mam_6180_mar_6180 = createMorphedHemisphere(p, 0.089875, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 1.67079632679, vec3(0.0559875, 0., -0.0221), 0.858, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.348775), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj mam_6180_apr_6180 = createMorphedHemisphere(p, 0.1929, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 2.19439510239, vec3(0.1497, 0., -0.1065), 0.858, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.348775), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj mam_6180_mai_6180 = createMorphedHemisphere(p, 0.575225, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 2.71799387799, vec3(0.4226625, 0., -0.2701), 0.858, vec4(vec3(0.19215686274509805, 0.6392156862745098, 0.32941176470588235), 1.0), 2.19439510239, vec3(0.6033875, 0., -0.348775), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj mam_4 = rayUnion(rayUnion(mam_6180_mar_6180, mam_6180_apr_6180), mam_6180_mai_6180);
obj jja_0120_jun_0120 = createMorphedHemisphere(p, 1.7232, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 3.24159265359, vec3(0.8616, 0., -0.0), 4.136825, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj jja_0120_jul_0120 = createMorphedHemisphere(p, 1.43315, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 3.76519142919, vec3(0.716575, 0., -0.0), 4.136825, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj jja_0120_aug_0120 = createMorphedHemisphere(p, 0.980475, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 4.28879020479, vec3(0.4902375, 0., -0.0), 4.136825, vec4(vec3(0.996078431372549, 0.8980392156862745, 0.8509803921568627), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj jja_0 = rayUnion(rayUnion(jja_0120_jun_0120, jja_0120_jul_0120), jja_0120_aug_0120);
obj jja_2140_jun_2140 = createMorphedHemisphere(p, 1.513525, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 3.24159265359, vec3(0.8616, 0., -0.209675), 3.857825, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.279), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj jja_2140_jul_2140 = createMorphedHemisphere(p, 1.41375, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 3.76519142919, vec3(0.716575, 0., -0.0194), 3.857825, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.279), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj jja_2140_aug_2140 = createMorphedHemisphere(p, 0.93055, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 4.28879020479, vec3(0.4902375, 0., -0.049925), 3.857825, vec4(vec3(0.9882352941176471, 0.6823529411764706, 0.5686274509803921), 1.0), 3.76519142919, vec3(2.0684125, 0., -0.279), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj jja_1 = rayUnion(rayUnion(jja_2140_jun_2140, jja_2140_jul_2140), jja_2140_aug_2140);
obj jja_4160_jun_4160 = createMorphedHemisphere(p, 1.2825, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 3.24159265359, vec3(0.8616, 0., -0.4407), 3.1289, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.007925), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj jja_4160_jul_4160 = createMorphedHemisphere(p, 1.0761, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 3.76519142919, vec3(0.716575, 0., -0.35705), 3.1289, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.007925), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj jja_4160_aug_4160 = createMorphedHemisphere(p, 0.7703, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 4.28879020479, vec3(0.4902375, 0., -0.210175), 3.1289, vec4(vec3(0.9372549019607843, 0.47058823529411764, 0.396078431372549), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.007925), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj jja_2 = rayUnion(rayUnion(jja_4160_jun_4160, jja_4160_jul_4160), jja_4160_aug_4160);
obj jja_6180_jun_6180 = createMorphedHemisphere(p, 0.921275, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 3.24159265359, vec3(0.8616, 0., -0.801925), 2.319275, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.81755), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj jja_6180_jul_6180 = createMorphedHemisphere(p, 0.8202, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 3.76519142919, vec3(0.716575, 0., -0.61295), 2.319275, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.81755), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj jja_6180_aug_6180 = createMorphedHemisphere(p, 0.5778, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 4.28879020479, vec3(0.4902375, 0., -0.402675), 2.319275, vec4(vec3(0.8901960784313725, 0.2627450980392157, 0.2235294117647059), 1.0), 3.76519142919, vec3(2.0684125, 0., -1.81755), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj jja_3 = rayUnion(rayUnion(jja_6180_jun_6180, jja_6180_jul_6180), jja_6180_aug_6180);
obj jja_8100_jun_8100 = createMorphedHemisphere(p, 0.87605, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 3.24159265359, vec3(0.8616, 0., -0.84715), 2.0843, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 3.76519142919, vec3(2.0684125, 0., -2.052525), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj jja_8100_jul_8100 = createMorphedHemisphere(p, 0.757875, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 3.76519142919, vec3(0.716575, 0., -0.675275), 2.0843, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 3.76519142919, vec3(2.0684125, 0., -2.052525), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj jja_8100_aug_8100 = createMorphedHemisphere(p, 0.450375, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 4.28879020479, vec3(0.4902375, 0., -0.5301), 2.0843, vec4(vec3(0.8666666666666667, 0.1568627450980392, 0.13725490196078433), 1.0), 3.76519142919, vec3(2.0684125, 0., -2.052525), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj jja_4 = rayUnion(rayUnion(jja_8100_jun_8100, jja_8100_jul_8100), jja_8100_aug_8100);
obj djf_2140_dez_0120 = createMorphedHemisphere(p, 0.172225, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 4.81238898038, vec3(0.0861125, 0., -0.0), 0.402175, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj djf_2140_jan_2140 = createMorphedHemisphere(p, 0.12555, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 5.33598775598, vec3(0.062775, 0., -0.0), 0.402175, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj djf_2140_feb_2140 = createMorphedHemisphere(p, 0.107825, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 5.85958653158, vec3(0.0539125, 0., -0.0), 0.402175, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0), between0and1, vec3(0.0001 + between0and1Sin * 0.0, 0., 0.));
obj djf_0 = rayUnion(rayUnion(djf_2140_dez_0120, djf_2140_jan_2140), djf_2140_feb_2140);
obj djf_0120_dez_2140 = createMorphedHemisphere(p, 0.1688, vec4(vec3(0.7411764705882353, 0.8431372549019608, 0.9058823529411765), 1.0), 4.81238898038, vec3(0.0861125, 0., -0.003425), 0.395275, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0069), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj djf_0120_jan_0120 = createMorphedHemisphere(p, 0.125, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 5.33598775598, vec3(0.062775, 0., -0.00055), 0.395275, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0069), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj djf_0120_feb_0120 = createMorphedHemisphere(p, 0.09805, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 5.85958653158, vec3(0.0539125, 0., -0.009775), 0.395275, vec4(vec3(0.9372549019607843, 0.9529411764705882, 1.0), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.0069), between0and1, vec3(0.0001 + between0and1Sin * 1.0, 0., 0.));
obj djf_1 = rayUnion(rayUnion(djf_0120_dez_2140, djf_0120_jan_0120), djf_0120_feb_0120);
obj djf_4160_dez_4160 = createMorphedHemisphere(p, 0.145525, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 4.81238898038, vec3(0.0861125, 0., -0.0267), 0.350175, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.052), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj djf_4160_jan_4160 = createMorphedHemisphere(p, 0.110375, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 5.33598775598, vec3(0.062775, 0., -0.015175), 0.350175, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.052), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj djf_4160_feb_4160 = createMorphedHemisphere(p, 0.094275, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 5.85958653158, vec3(0.0539125, 0., -0.01355), 0.350175, vec4(vec3(0.4196078431372549, 0.6823529411764706, 0.8392156862745098), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.052), between0and1, vec3(0.0001 + between0and1Sin * 2.0, 0., 0.));
obj djf_2 = rayUnion(rayUnion(djf_4160_dez_4160, djf_4160_jan_4160), djf_4160_feb_4160);
obj djf_8100_dez_8100 = createMorphedHemisphere(p, 0.129725, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 4.81238898038, vec3(0.0861125, 0., -0.0425), 0.318725, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.08345), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj djf_8100_jan_8100 = createMorphedHemisphere(p, 0.09885, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 5.33598775598, vec3(0.062775, 0., -0.0267), 0.318725, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.08345), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj djf_8100_feb_8100 = createMorphedHemisphere(p, 0.09015, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 5.85958653158, vec3(0.0539125, 0., -0.017675), 0.318725, vec4(vec3(0.11372549019607843, 0.41568627450980394, 0.6784313725490196), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.08345), between0and1, vec3(0.0001 + between0and1Sin * 3.0, 0., 0.));
obj djf_3 = rayUnion(rayUnion(djf_8100_dez_8100, djf_8100_jan_8100), djf_8100_feb_8100);
obj djf_6180_dez_6180 = createMorphedHemisphere(p, 0.1243, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 4.81238898038, vec3(0.0861125, 0., -0.047925), 0.303225, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.09895), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj djf_6180_jan_6180 = createMorphedHemisphere(p, 0.095725, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 5.33598775598, vec3(0.062775, 0., -0.029825), 0.303225, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.09895), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj djf_6180_feb_6180 = createMorphedHemisphere(p, 0.0832, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 5.85958653158, vec3(0.0539125, 0., -0.024625), 0.303225, vec4(vec3(0.19215686274509805, 0.5098039215686274, 0.7411764705882353), 1.0), 5.33598775598, vec3(0.2010875, 0., -0.09895), between0and1, vec3(0.0001 + between0and1Sin * 4.0, 0., 0.));
obj djf_4 = rayUnion(rayUnion(djf_6180_dez_6180, djf_6180_jan_6180), djf_6180_feb_6180);
return rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(son_0, son_1), son_2), son_3), son_4), mam_0), mam_1), mam_2), mam_3), mam_4), jja_0), jja_1), jja_2), jja_3), jja_4), djf_0), djf_1), djf_2), djf_3), djf_4);

    /*vec3 p1 = rayRotationZ(p, PI/6.);
    p1 = rayTranslation(p1, vec3(-2., 0., 0.));
    vec3 p2 = rayRotationZ(p, 2.*PI/6.);
    p2 = rayTranslation(p2, vec3(-2., 0., 0.));
    vec3 p3 = rayRotationZ(p, 3.*PI/6.);
    p3 = rayTranslation(p3, vec3(-2., 0., 0.));

    obj m1;
    m1.distance = rayHemisphere(rayTranslation(p1, vec3(0.0001 + between0and1Sin, 0., 0.)), 0.5) * between0and1 + rayHemisphere(rayTranslation(p2, vec3(0.0001 + between0and1Sin, 0., 0.)), 0.8) * (1. - between0and1);
    m1.color = vec4(vec3(255.,0.,0.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 0.) / 255., 1.) * (1. - between0and1);

    obj m1 = createMorphedHemisphere(p, 0.5, vec4(vec3(255.,0.,0.) / 255., 1.), PI/6.,
                                        0.8, vec4(vec3(255.,255., 0.) / 255., 1.), 2.*PI/6.,
                                        between0and1, vec3(0.0001 + between0and1Sin, 0., 0.));

    obj m2;
    m2.distance = rayHemisphere(rayTranslation(p2, vec3(0.0001 + between0and1Sin, 0, 0.)), 0.5) * between0and1 + rayHemisphere(rayTranslation(p2, vec3(0.0001 + between0and1Sin, 0., 0.)), 0.8) * (1. - between0and1);
    m2.color = vec4(vec3(0., 255.,0.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 0.) / 255., 1.) * (1. - between0and1);

    obj m3;
    m3.distance = rayHemisphere(rayTranslation(p3, vec3(0.0001 + between0and1Sin, 0., 0.)), 0.5) * between0and1 + rayHemisphere(rayTranslation(p2, vec3(0.0001 + between0and1Sin, 0., 0.)), 0.8) * (1. - between0and1);
    m3.color = vec4(vec3(0.,0.,255.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 0.) / 255., 1.) * (1. - between0and1);

    obj a = rayUnion(rayUnion(m1, m2), m3);

    m1.distance = rayHemisphere(p1, 0.6) * between0and1 + rayHemisphere(p2, 0.9) * (1. - between0and1);
    m1.color = vec4(vec3(255.,125.,0.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 125.) / 255., 1.) * (1. - between0and1);

    m2.distance = rayHemisphere(p2, 0.6) * between0and1 + rayHemisphere(p2, 0.9) * (1. - between0and1);
    m2.color = vec4(vec3(0., 255.,125.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 125.) / 255., 1.) * (1. - between0and1);

    m3.distance = rayHemisphere(p3, 0.6) * between0and1 + rayHemisphere(p2, 0.9) * (1. - between0and1);
    m3.color = vec4(vec3(125.,0.,255.) / 255., 1.) * between0and1 + vec4(vec3(255.,255., 125.) / 255., 1.) * (1. - between0and1);

    obj b = rayUnion(rayUnion(m1, m2), m3);

    return rayUnion(b, a); */
}

obj mapRevolvingHemispheres(vec3 p) {
    p = rayTranslation(p, vec3(0., 7., 7.));
    p = opTx( p, Rot4X( PI/4.0 ) );

    p = rayRotationZ(p, -czm_frameNumber * 0.01);

    vec3 p1 = rayTranslation(p, vec3(-4. + 8.045 / 20., 0, 0));

	obj mam_6180;
	mam_6180.distance = rayHemisphere(rayTranslationZ(p1, 5.72 / 10. - 8.045 / 10.), 5.72 / 10.);
	mam_6180.color = vec4(49.,163.,84., 255.)/ 255.; //#31a354

    obj mam_8199;
    mam_8199.distance = rayHemisphere(rayTranslationZ(p1, 6.115 / 10. - 8.045 / 10.), 6.115 / 10.);
    mam_8199.color = vec4(0.,109.,44., 255.)/ 255.; //#006d2c

	obj mam_2140;
	mam_2140.distance = rayHemisphere(rayTranslationZ(p1, 7.533 / 10. - 8.045 / 10.), 7.533 / 10.);
	mam_2140.color = vec4(186.,228.,179., 255.)/ 255.; //#bae4b3

	obj mam_0120;
	mam_0120.distance = rayHemisphere(rayTranslationZ(p1, 7.765 / 10. - 8.045 / 10.), 7.765 / 10.);
	mam_0120.color = vec4(237.,248.,233., 255.)/ 255.; //#edf8e9

	obj mam_4160;
	mam_4160.distance = rayHemisphere(p1, 8.045 / 10.);
	mam_4160.color = vec4(116.,196.,118., 255.)/ 255.; //#74c476

    obj a = rayUnion(rayUnion(rayUnion(rayUnion(mam_4160, mam_0120), mam_2140), mam_8199), mam_6180);

    vec3 p2 = rayRotationZ(p, PI/2.);
    p2 = rayTranslation(p2, vec3(-4. + 27.579 / 20., 0, 0));

	obj jja_8199 ;
	jja_8199.distance = rayHemisphere(rayTranslationZ(p2, 14.018 / 10. - 27.579 / 10.), 14.018 / 10.);
	jja_8199.color = vec4(214.,13.,13., 255.)/ 255.; //#d60d0d

	obj jja_6180;
	jja_6180.distance = rayHemisphere(rayTranslationZ(p2, 15.462 / 10. - 27.579 / 10.), 15.462 / 10.);
	jja_6180.color = vec4(227.,67.,57., 255.)/ 255.; //#e34339

	obj jja_4160;
	jja_4160.distance = rayHemisphere(rayTranslationZ(p2, 20.859 / 10. - 27.579 / 10.), 20.859 / 10.);
	jja_4160.color = vec4(239.,135.,101., 255.)/ 255.; //#ef8765

    obj jja_2140;
    jja_2140.distance = rayHemisphere(rayTranslationZ(p2, 25.719 / 10. - 27.579 / 10.), 25.719 / 10.);
    jja_2140.color = vec4(252.,174.,145., 255.)/ 255.; //#fcae91

	obj jja_0120;
	jja_0120.distance = rayHemisphere(p2, 27.579 / 10.);
	jja_0120.color = vec4(254.,229.,217., 255.)/ 255.; //#fee5d9

    obj b = rayUnion(rayUnion(rayUnion(rayUnion(jja_0120, jja_2140), jja_4160), jja_6180), jja_8199);

    vec3 p3 = rayRotationZ(p, PI);
    p3 = rayTranslation(p3, vec3(-4. + 8.98 / 20., 0, 0));

	obj son_6180;
	son_6180.distance = rayHemisphere(rayTranslationZ(p3, 5.828 / 10. - 8.98 / 10.), 5.828 / 10.);
	son_6180.color = vec4(254.,179.,101., 255.)/ 255.; //#feb365

	obj son_8199;
	son_8199.distance = rayHemisphere(rayTranslationZ(p3, 6.302 / 10. - 8.98 / 10.), 6.302 / 10.);
	son_8199.color = vec4(254.,153.,41., 255.)/ 255.; //#fe9929

	obj son_4160;
	son_4160.distance = rayHemisphere(rayTranslationZ(p3, 7.767 / 10. - 8.98 / 10.), 7.767 / 10.);
	son_4160.color = vec4(255.,204.,131., 255.)/ 255.; //#ffcc83

    obj son_0120;
    son_0120.distance = rayHemisphere(rayTranslationZ(p3, 8.755 / 10. - 8.98 / 10.), 8.755 / 10.);
    son_0120.color = vec4(255.,255.,221., 255.)/ 255.; //#ffffdd

	obj son_2140;
	son_2140.distance = rayHemisphere(p3, 8.98 / 10.);
	son_2140.color = vec4(255.,230.,176., 255.)/ 255.; //#ffe6b0

    obj c = rayUnion(rayUnion(rayUnion(rayUnion(son_2140, son_0120), son_4160), son_8199), son_6180);

    vec3 p4 = rayRotationZ(p, 1.5*PI);
    p4 = rayTranslation(p4, vec3(-4. + 2.681 / 20., 0, 0));

	obj djf_6180;
	djf_6180.distance = rayHemisphere(rayTranslationZ(p4, 2.022 / 10. - 2.681 / 10.), 2.022 / 10.);
	djf_6180.color = vec4(49.,130.,189., 255.)/ 255.; //#3182bd

	obj djf_8199;
	djf_8199.distance = rayHemisphere(rayTranslationZ(p4, 2.085 / 10. - 2.681 / 10.), 2.085 / 10.);
	djf_8199.color = vec4(8.,81.,156., 255.)/ 255.; //#08519c

	obj djf_4160;
	djf_4160.distance = rayHemisphere(rayTranslationZ(p4, 2.335 / 10. - 2.681 / 10.), 2.335 / 10.);
	djf_4160.color = vec4(107.,174.,214., 255.)/ 255.; //#6baed6

    obj djf_0120;
    djf_0120.distance = rayHemisphere(rayTranslationZ(p4, 2.635 / 10. - 2.681 / 10.), 2.635 / 10.);
    djf_0120.color = vec4(239.,243.,255., 255.)/ 255.; //#eff3ff

	obj djf_2140;
	djf_2140.distance = rayHemisphere(p4, 2.681 / 10.);
	djf_2140.color = vec4(189.,215.,231., 255.) / 255.; //#bdd7e7

    obj d = rayUnion(rayUnion(rayUnion(rayUnion(djf_2140, djf_0120), djf_4160), djf_8199), djf_6180);

    return rayUnion(rayUnion(rayUnion(a, b), c), d);
}


obj mapDonutCutWithPole(vec3 p) { //DonutCutWithPole
    float lat = -10.0 * PI / 180.;
    float lon = 10.0 * PI / 180.;
    vec3 pos = vec3(7000000, 0, 0);
    pos = opTx(opTx(pos, Rot4Y(lat)), Rot4Z(lon)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Y( -PI/2.0 - lat) );
    p = opTx( p, Rot4Z( PI) );

    float between0And1 = cos(czm_frameNumber * 0.01 * PI) * 0.5 + 0.5;
    float zeroOrOne = sign(sin(czm_frameNumber * 0.01 * PI)) * 0.5 + 0.5;

    obj rayTorusSegment1;
    rayTorusSegment1.distance = rayTorusSegment(p, vec2(1.5, 0.5), 0., PI/3.);
    rayTorusSegment1.color = vec4(vec3(116.,169.,207.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment2;
    rayTorusSegment2.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI/3., PI/3. + PI/4.);
    rayTorusSegment2.color = vec4(vec3(43.,140.,190.) / 255., 1.);

    obj rayTorusSegment3;
    rayTorusSegment3.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI/3. + PI/4., PI/3. + PI/4. + 5.*PI/12.);
    rayTorusSegment3.color = vec4(vec3(4.,90.,141.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment4;
    rayTorusSegment4.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI, PI + 2.*PI/3.);
    rayTorusSegment4.color = vec4(vec3(251.,180.,185.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment5;
    rayTorusSegment5.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI + 2.*PI/3., PI + 2.*PI/3. + PI/4.);
    rayTorusSegment5.color = vec4(vec3(247.,104.,161.) / 255., 1.);

    obj rayTorusSegment6;
    rayTorusSegment6.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI + 2.*PI/3. + PI/4., 2.*PI);
    rayTorusSegment6.color = vec4(vec3(174.,1.,126.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayBox1;
    rayBox1.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3.), vec3(between0And1 + 0.405, 0, 0)), vec3(0.6, 0.05, 1.));
    rayBox1.color = vec4(1., 1., 1., 1.);

    obj rayBox2;
    rayBox2.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3. + PI/4.), vec3(between0And1 + 0.405, 0, 0)), vec3(0.6, 0.05, 1.));
    rayBox2.color = vec4(1., 1., 1., 1.);

    obj rayBox3;
    rayBox3.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3.), vec3(between0And1 + 0.405, 0, 0)), vec3(0.6, 0.05, 1.));
    rayBox3.color = vec4(1., 1., 1., 1.);

    obj rayBox4;
    rayBox4.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3. + PI/4.), vec3(between0And1 + 0.405, 0, 0)), vec3(0.6, 0.05, 1.));
    rayBox4.color = vec4(1., 1., 1., 1.);

    obj rayBoxCutter1;
    rayBoxCutter1.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3.), vec3(1.005 + between0And1, 0, 0)), vec3(0.05, 0.05, 1.));
    rayBoxCutter1.color = vec4(1., 1., 1., 1.);

    obj rayBoxCutter2;
    rayBoxCutter2.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3. + PI/4.), vec3(1.005 + between0And1, 0, 0)), vec3(0.05, 0.05, 1.));
    rayBoxCutter2.color = vec4(1., 1., 1., 1.);

    obj rayBoxCutter3;
    rayBoxCutter3.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3.), vec3(1.005 + between0And1, 0, 0)), vec3(0.05, 0.05, 1.));
    rayBoxCutter3.color = vec4(1., 1., 1., 1.);

    obj rayBoxCutter4;
    rayBoxCutter4.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3. + PI/4.), vec3(1.005 + between0And1, 0, 0)), vec3(0.05, 0.05, 1.));
    rayBoxCutter4.color = vec4(1., 1., 1., 1.);

    obj a = rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayTorusSegment1, rayTorusSegment2), rayTorusSegment3), rayTorusSegment4), rayTorusSegment5), rayTorusSegment6);
    obj a1 = rayUnion(rayUnion(rayUnion(rayUnion(rayBoxCutter1, rayBoxCutter2), rayBoxCutter3), rayBoxCutter4), a);
    obj b = raySubtraction(rayUnion(rayUnion(rayUnion(rayBox1, rayBox2), rayBox3), rayBox4), a1);
    return b;
}


obj mapDonutCut(vec3 p) {
    float lat = -10.0 * PI / 180.;
    float lon = 10.0 * PI / 180.;
    vec3 pos = vec3(7000000, 0, 0);
    pos = opTx(opTx(pos, Rot4Y(lat)), Rot4Z(lon)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Y( -PI/2.0 - lat) );
    p = opTx( p, Rot4Z( PI) );

    float between0And1 = cos(czm_frameNumber * 0.01 * PI) * 0.5 + 0.5;
    float zeroOrOne = sign(sin(czm_frameNumber * 0.01 * PI)) * 0.5 + 0.5;

    obj rayTorusSegment1;
    rayTorusSegment1.distance = rayTorusSegment(p, vec2(1.5, 0.5), 0., PI/3.);
    rayTorusSegment1.color = vec4(vec3(116.,169.,207.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment2;
    rayTorusSegment2.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI/3., PI/3. + PI/4.);
    rayTorusSegment2.color = vec4(vec3(43.,140.,190.) / 255., 1.);

    obj rayTorusSegment3;
    rayTorusSegment3.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI/3. + PI/4., PI/3. + PI/4. + 5.*PI/12.);
    rayTorusSegment3.color = vec4(vec3(4.,90.,141.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment4;
    rayTorusSegment4.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI, PI + 2.*PI/3.);
    rayTorusSegment4.color = vec4(vec3(251.,180.,185.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTorusSegment5;
    rayTorusSegment5.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI + 2.*PI/3., PI + 2.*PI/3. + PI/4.);
    rayTorusSegment5.color = vec4(vec3(247.,104.,161.) / 255., 1.);

    obj rayTorusSegment6;
    rayTorusSegment6.distance = rayTorusSegment(p, vec2(1.5, 0.5), PI + 2.*PI/3. + PI/4., 2.*PI);
    rayTorusSegment6.color = vec4(vec3(174.,1.,126.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayBox1;
    rayBox1.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3.), vec3(between0And1 + 0.405, 0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox1.color = vec4(vec3(116.,169.,207.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayBox2;
    rayBox2.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3.), vec3(between0And1 + 0.405, -0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox2.color = vec4(vec3(43.,140.,190.) / 255., 1.);

    obj rayBox3;
    rayBox3.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3. + PI/4.), vec3(between0And1 + 0.405, 0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox3.color = vec4(vec3(43.,140.,190.) / 255., 1.);

    obj rayBox4;
    rayBox4.distance = rayBox(rayTranslation(rayRotationZ(p, PI/3. + PI/4.), vec3(between0And1 + 0.405, -0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox4.color = vec4(vec3(4.,90.,141.) / 255. * zeroOrOne + vec3(43.,140.,190.) / 255. * (1. - zeroOrOne), 1.);

    obj rayBox5;
    rayBox5.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3.), vec3(between0And1 + 0.405, 0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox5.color = vec4(vec3(251.,180.,185.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayBox6;
    rayBox6.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3.), vec3(between0And1 + 0.405, -0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox6.color = vec4(vec3(247.,104.,161.) / 255., 1.);

    obj rayBox7;
    rayBox7.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3. + PI/4.), vec3(between0And1 + 0.405, 0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox7.color = vec4(vec3(247.,104.,161.) / 255., 1.);

    obj rayBox8;
    rayBox8.distance = rayBox(rayTranslation(rayRotationZ(p, PI + 2.*PI/3. + PI/4.), vec3(between0And1 + 0.405, -0.025, 0)), vec3(0.6, 0.05, 1.));
    rayBox8.color = vec4(vec3(174.,1.,126.) / 255. * zeroOrOne + vec3(247.,104.,161.) / 255. * (1. - zeroOrOne), 1.);

    obj rayTori = rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayTorusSegment1, rayTorusSegment2), rayTorusSegment3), rayTorusSegment4), rayTorusSegment5), rayTorusSegment6);
    obj rayBoxes = rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(rayBox1, rayBox2), rayBox3), rayBox4), rayBox5), rayBox6), rayBox7), rayBox8);
    return raySubtraction(rayBoxes, rayTori);
}

obj mapPyramidFrustumTransparency(vec3 p) {
    float lat = -11.304977 * PI / 180.;
    float lon = -59.727864 * PI / 180.;
    vec3 pos = vec3(7000000, 0, 0);
    pos = opTx(opTx(pos, Rot4Y(-lat)), Rot4Z(lon)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Y( -PI/2.0 + lat) );
    p = opTx( p, Rot4X( lon) );
    p = opTx( p, Rot4Z( PI/4.0 ));

    float between0And1 = cos(czm_frameNumber * 0.01 * PI) * 0.5 + 0.5;
    float alpha = between0And1; //0.25;

    obj pyramidFrustum1;
    pyramidFrustum1.distance = rayPyramidFrustum(p, 0.5, 1., 0.2);
    pyramidFrustum1.color = vec4(vec3(237., 248., 233.) / 255., alpha);

    obj pyramidFrustum2;
    pyramidFrustum2.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 0.4)), 1., 0.5, 0.2);
    pyramidFrustum2.color = vec4(vec3(190., 232., 190.)/ 255., alpha);
    //pyramidFrustum2.color = vec4(vec3(1., 1., 1.), 0.10);

    obj pyramidFrustum3;
    pyramidFrustum3.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 0.8)), 0.5, 0.75, 0.2);
    pyramidFrustum3.color = vec4(vec3(142., 217., 146.) / 255., 1.);

    obj pyramidFrustum4;
    pyramidFrustum4.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 1.2)), 0.75, 0.9, 0.2);
    pyramidFrustum4.color = vec4(vec3(95., 201., 103.)/ 255., alpha);

    obj pyramidFrustum5;
    pyramidFrustum5.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 1.6)), 0.9, 0.6, 0.2);
    pyramidFrustum5.color = vec4(vec3(47., 186., 59.) / 255., alpha);

    obj pyramidFrustum6;
    pyramidFrustum6.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 2.0)), 0.6, 0.2, 0.2);
    pyramidFrustum6.color = vec4(vec3(0., 170., 16) / 255., alpha);

    return rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(pyramidFrustum1, pyramidFrustum2), pyramidFrustum3), pyramidFrustum4), pyramidFrustum5), pyramidFrustum6);
}



obj mapScaledPyramidFrustums(vec3 p) {
    float lat = -11.304977 * PI / 180.;
    float lon = -59.727864 * PI / 180.;
    vec3 pos = vec3(7000000, 0, 0);
    pos = opTx(opTx(pos, Rot4Y(-lat)), Rot4Z(lon)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Y( -PI/2.0 + lat) );
    p = opTx( p, Rot4X( lon) );
    p = opTx( p, Rot4Z( PI/4.0 ));

    float top = cos(czm_frameNumber * 0.01 * PI) * 0.2 + 0.2;
    float bottom = cos(czm_frameNumber * 0.01 * PI + PI) * 0.2 + 0.2;

    obj pyramidFrustum1;
    pyramidFrustum1.distance = rayPyramidFrustum(p, 0.5 + top, 1. + bottom, 0.2);
    pyramidFrustum1.color = vec4(vec3(237., 248., 233.) / 255., 1.);

    obj pyramidFrustum2;
    pyramidFrustum2.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 0.4)), 1. + bottom, 0.5 + top, 0.2);
    pyramidFrustum2.color = vec4(vec3(190., 232., 190.) / 255., 1.);

    obj pyramidFrustum3;
    pyramidFrustum3.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 0.8)), 0.5 + top, 0.75 + bottom, 0.2);
    pyramidFrustum3.color = vec4(vec3(142., 217., 146.) / 255., 1.);

    obj pyramidFrustum4;
    pyramidFrustum4.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 1.2)), 0.75 + bottom, 0.9 + top, 0.2);
    pyramidFrustum4.color = vec4(vec3(95., 201., 103.) / 255., 1.);

    obj pyramidFrustum5;
    pyramidFrustum5.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 1.6)), 0.9 + top, 0.6 + bottom, 0.2);
    pyramidFrustum5.color = vec4(vec3(47., 186., 59.) / 255., 1.);

    obj pyramidFrustum6;
    pyramidFrustum6.distance = rayPyramidFrustum(rayTranslation(p, vec3(0., 0., 2.0)), 0.6 + bottom, 0.2 + top, 0.2);
    pyramidFrustum6.color = vec4(vec3(0., 170., 16) / 255., 1.);

    return rayUnion(rayUnion(rayUnion(rayUnion(rayUnion(pyramidFrustum1, pyramidFrustum2), pyramidFrustum3), pyramidFrustum4), pyramidFrustum5), pyramidFrustum6);
}

obj map(vec3 p) { //ExtrudedPieSegment
    float lat = 40.415342 * PI / 180.;
    float lon = (-96.903453 + 90.) * PI / 180.;
    vec3 pos = vec3(0, -7000000, 0);
    pos = opTx(opTx(pos, Rot4Z(lon)), Rot4X(-lat)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Z( - lon  ) );
    p = opTx( p, Rot4X( -PI/2.0 + lat) );

    obj pieSegment1;
    pieSegment1.distance = rayPieSegment(p, radians(0.), radians(30.), cos(czm_frameNumber * 0.015 * PI) * 0.5 + 1., 0.);
    pieSegment1.color = vec4(vec3(251.,180.,174.) / 255., 1.);

    obj pieSegment2;
    pieSegment2.distance = rayPieSegment(p, radians(30.), radians(75.), cos((czm_frameNumber + 75.) * 0.015 * PI) * 0.5 + 1., 0.);
    pieSegment2.color = vec4(vec3(179.,205.,227.) / 255., 1.);

    obj pieSegment3;
    pieSegment3.distance = rayPieSegment(p, radians(75.), radians(150.), cos((czm_frameNumber + 25.) * 0.015 * PI) * 0.5 + 1., 0.);
    pieSegment3.color = vec4(vec3(204.,235.,197.) / 255., 1.);

    obj pieSegment4;
    pieSegment4.distance = rayPieSegment(p, radians(150.), radians(340.), cos((czm_frameNumber + 50.) * 0.015 * PI) * 0.5 + 1., 0.);
    pieSegment4.color = vec4(vec3(222.,203.,228.) / 255., 1.);

    obj pieSegment5;
    pieSegment5.distance = rayPieSegment(p, radians(340.), radians(360.), cos((czm_frameNumber + 33.) * 0.015 * PI) * 0.5 + 1., 0.);
    pieSegment5.color = vec4(vec3(254.,217.,166.) / 255., 1.);

    return rayUnion(rayUnion(rayUnion(rayUnion(pieSegment1, pieSegment2), pieSegment3), pieSegment4), pieSegment5);
}


obj mapExplodedPieSegments(vec3 p) {
    float lat = 40.415342 * PI / 180.;
    float lon = (-96.903453 + 90.) * PI / 180.;
    vec3 pos = vec3(0, -7000000, 0);
    pos = opTx(opTx(pos, Rot4Z(lon)), Rot4X(-lat)) * MINIFIER;

    p = rayTranslation(p, pos);
    p = opTx( p, Rot4Z( - lon  ) );
    p = opTx( p, Rot4X( -PI/2.0 + lat) );

    float explosionTranslation = cos(czm_frameNumber * 0.05) * 0.5 + 0.5;

    obj pieSegment1;
    pieSegment1.distance = rayPieSegment(p, radians(0.), radians(30.), 1.75, explosionTranslation);
    pieSegment1.color = vec4(vec3(251.,180.,174.) / 255., 1.);

    obj pieSegment2;
    pieSegment2.distance = rayPieSegment(p, radians(30.), radians(75.), 1.25, explosionTranslation);
    pieSegment2.color = vec4(vec3(179.,205.,227.) / 255., 1.);

    obj pieSegment3;
    pieSegment3.distance = rayPieSegment(p, radians(75.), radians(150.), 1.5, explosionTranslation);
    pieSegment3.color = vec4(vec3(204.,235.,197.) / 255., 1.);

    obj pieSegment4;
    pieSegment4.distance = rayPieSegment(p, radians(150.), radians(340.), 1., explosionTranslation);
    pieSegment4.color = vec4(vec3(222.,203.,228.) / 255., 1.);

    obj pieSegment5;
    pieSegment5.distance = rayPieSegment(p, radians(340.), radians(360.), 2., explosionTranslation);
    pieSegment5.color = vec4(vec3(254.,217.,166.) / 255., 1.);

    return rayUnion(rayUnion(rayUnion(rayUnion(pieSegment1, pieSegment2), pieSegment3), pieSegment4), pieSegment5);
}



float sdCappedCylinder( vec3 p, vec2 h ) {
    vec2 d = abs(vec2(length(p.xy),p.z)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}


float sdTriPrism( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
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





vec3 calcNormal(vec3 intersectionPoint) {
    vec3 eps = vec3( 0.001, 0.0, 0.0 );
    vec3 nor = vec3(
        map(intersectionPoint+eps.xyy).distance - map(intersectionPoint-eps.xyy).distance,
        map(intersectionPoint+eps.yxy).distance - map(intersectionPoint-eps.yxy).distance,
        map(intersectionPoint+eps.yyx).distance - map(intersectionPoint-eps.yyx).distance
    );
    return normalize(nor);
}

void main() {
    float u = gl_FragCoord.x * 2.0 / czm_viewport.z - 1.0;
    float v = gl_FragCoord.y * 2.0 / czm_viewport.w - 1.0;

    vec3 rayDirection = normalize(cameraForward + cameraRight * u * tanCameraFovHalf + cameraUp * v * tanCameraFovYHalf);
    //same as:
    //vec4 rayDirectionNDC = vec4(u, v, 0, 1);
    //vec4 rayDirectionWC = czm_inverseView * czm_inverseProjection * rayDirectionNDC;
    //vec3 rayDirectionWCDivided = vec3(rayDirectionWC.xyz / rayDirectionWC.w);
    //vec3 rayDirection = normalize(rayDirectionWCDivided - czm_viewerPositionWC);

    float maxDistance = czm_currentFrustum.y * 2. * MINIFIER;
    float d;
    float t = 0.0;
    float resultAlpha = 0.0;
    vec3 resultColor = vec3(0, 0, 0);
    int steps = MAX_STEPS;

    /*bool intersects = checkIfRayIntersectsBoundingSpheres(czm_viewerPositionWC * MINIFIER, rayDirection);

    if (!intersects)
        discard; */

    for (float i = 0.; i < 1000.; i++) {
        vec3 p = (czm_viewerPositionWC * MINIFIER) + rayDirection * t;
        obj result = map(p);
        d = result.distance; //abs(result.distance) for transparency

        if (d < EPSILON) {
            float distance = t * (1. / MINIFIER);
            float cosineAngle = dot(cameraForward, rayDirection);
            float z = cosineAngle * distance;
            float depth = depthA - depthB/z;
            //same as:
            //vec4 intersectionPoint = vec4(czm_viewerPositionWC + rayDirection * distance, 1.);
            //vec4 projectedIntersectionPoint = (czm_projection * czm_view * intersectionPoint);
            //float depth = projectedIntersectionPoint.z / projectedIntersectionPoint.w

            float fragDepth = (1. + depth) * 0.5;

            gl_FragDepthEXT = fragDepth;
            //resultAlpha = 1.;
            //resultColor = result.color.xyz;

            if (result.color.w > 0.) {
                vec3 normal = calcNormal(p);
                float diffuseFactor = abs(dot(normal, rayDirection));
                vec3 color = result.color.xyz * diffuseFactor;
                //vec3 color = result.color.xyz;

                float color_add_factor;
                if (resultAlpha + result.color.w >= 0.9999) {
                    color_add_factor = 1. - resultAlpha;
                    resultAlpha = 1.;
                } else {
                    resultAlpha += result.color.w;
                    color_add_factor = result.color.w / resultAlpha;
                }

                resultColor = resultColor * (1. - color_add_factor) + color * color_add_factor;

                //czm_pickColor = vec4(1./255., 0., 0., 1.);

                if (resultAlpha >= 0.9999) {
                    break;
                }
            }

            t += EPSILON * 20.;
            steps = MAX_STEPS * 2;
        }

        t += d;
        steps--;

        if (steps == 0 || t > maxDistance) {
            break;
        }
    }

    if (resultAlpha > 0.) {
        gl_FragColor = vec4(resultColor, resultAlpha);
    } else {
        discard;
    }
}
