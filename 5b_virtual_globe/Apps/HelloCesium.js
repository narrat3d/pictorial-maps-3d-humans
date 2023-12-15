/*global require*/
require({
    baseUrl : '../Source',
    packages : [{
    }, {
        name : 'Source',
        location : '.'
    }]
}, [
    'Source/Cesium',
    'Source/Renderer/Texture',
    'Source/Renderer/PixelDatatype',
    'Source/Core/PixelFormat',
    'Source/Renderer/Sampler',
    'Source/Renderer/TextureMagnificationFilter',
    'Source/SpeechBubble/SpeechBubbleLayer'
],  function(
    Cesium, Texture, PixelDatatype, PixelFormat, Sampler, TextureMagnificationFilter, SpeechBubbleLayer) {
    'use strict';

    /*
     Matterhorn:
     View: 7.6586254119667805, 45.97645860999024, 10000000
     Figure: 7.656, 45.97645860999024, 4500 (Rotation x: -PI/4)
     Speechbubble: 7.657, 45.97645860999024, 5200

     Etzel:
     model rotation: {x: -Math.PI/4, y: -Math.PI / 4, z: 0.}
     positionProperty and route
     shader: walkingAnimation, walkingAnimationPoint, walkingAnimationNormal
     MINIFIER = 1. / 1.

     Football:
     position : new Cesium.Cartesian3(-20000000, -7500000, -10000000)
     modelRotation: {x: 0, y: Math.PI/4, z: 0.}
     speechBubbleLayer.add("What is the slogan of the EURO 2024 in Germany?\n\na) Among friends\nb) Expect emotions\nc) United by football", new Cesium.Cartesian3(-10000000, -5000000, 30000000)); //Cesium.Cartesian3.fromDegrees(0, 0, 0)
     const float MINIFIER = 1. / 250000.

     Gardening:
     https://www.businesslocationcenter.de/berlin3d-downloadportal/#/export KMZ with textures, use local coordinates
     https://github.com/KhronosGroup/COLLADA2GLTF/releases convert to GLTF -f infile -o outfile -e

     figures/Antoine_Corbineau_Metropolitan_Eurostar_Bruxelles_Brugman_map
     position : Cesium.Cartesian3.fromDegrees(13.572555052306038, 52.53968537527502, 30),
     modelRotation: {x: -Math.PI/4, y: 0, z: 0},
     buildings = var origin = Cesium.Cartesian3.fromDegrees(13.571337883952092, 52.541684834548064, -5);
     */

    //In order for CodeMirror auto-complete to work, Cesium needs to be defined as a global.
    window.Cesium = Cesium;

    Cesium.BingMapsApi.defaultKey = '<insert your key here>';

    const figureFolder = "figures/Antoine_Corbineau_Metropolitan_Eurostar_Bruxelles_Brugman_map";

    const POSE_POINTS = {
        'right_ankle': 0,
        'right_knee': 1,
        'right_hip': 2,
        'left_hip': 3,
        'left_knee': 4,
        'left_ankle': 5,
        'pelvis': 6,
        'root': 7,
        'neck': 8,
        'head': 9,
        'right_wrist': 10,
        'right_elbow': 11,
        'right_shoulder': 12,
        'left_shoulder': 13,
        'left_elbow': 14,
        'left_wrist': 15,
        'right_foot': 16,
        'left_foot': 17,
        'right_middle1': 18,
        'left_middle1': 19,
        'eyes': 20
    }

    const BONES = {
        "torso": ["right_shoulder", "left_shoulder", "left_hip", "right_hip"],
        "head": ["head"], //, "eyes"
        "right_upper_arm": ["right_shoulder", "right_elbow"],
        "right_lower_arm": ["right_elbow", "right_wrist"],
        "right_hand": ["right_wrist"], //, "right_middle1"
        "right_upper_leg": ["right_hip", "right_knee"],
        "right_lower_leg": ["right_knee", "right_ankle"],
        "right_foot": ["right_ankle"], //, "right_foot"
        "left_upper_leg": ["left_hip", "left_knee"],
        "left_lower_leg": ["left_knee", "left_ankle"],
        "left_foot": ["left_ankle"], //, "left_foot"
        "left_upper_arm": ["left_shoulder", "left_elbow"],
        "left_lower_arm": ["left_elbow", "left_wrist"],
        "left_hand": ["left_wrist"] //, "left_middle1"
    }

    const VIEWS = ["front_cropped", "left", "back", "right", "front"];

    const body_size = 300;
    const scaling_factor = 0.54;
    const adjusted_body_size = 308;

    var imageryProviders = Cesium.createDefaultImageryProviderViewModels();

    const viewer = new Cesium.Viewer('cesiumContainer', {
        imageryProviderViewModels: imageryProviders,
        selectedImageryProviderViewModel: imageryProviders[10],
    }); // {globe: false}
    viewer.extend(Cesium.viewerCesiumInspectorMixin);

    const scene = viewer.scene;
    const globe = scene.globe;
    //needed that the z-buffer works correctly
    globe.depthTestAgainstTerrain = true;

    //set camera to look at 0, 0 lat lon
    viewer.camera.setView({
        destination : Cesium.Cartesian3.fromDegrees(8.54087402600246, 47.364, 250),
        orientation: {
            heading: 0,
            pitch: Cesium.Math.toRadians(-30),
            roll: 0
        }
    });

    /*
    const startTime = Cesium.JulianDate.fromIso8601("2023-11-11T11:00:00+01");
    const stopTime = Cesium.JulianDate.fromIso8601("2023-11-11T13:00:00+01");

    const clock = viewer.clock;

    const clockProperties = {
        startTime: startTime,
        currentTime: startTime,
        stopTime: stopTime,
        clockRange: Cesium.ClockRange.CLAMPED,
        shouldAnimate: true,
        multiplier: 10,
    };

    for (const [key, value] of Object.entries(clockProperties)) {
        clock[key] = value;
    }
     */

    /*
    //need to set req.headers.Authorization in server.js (copy from Cesium Ion web developer tools)
    var cesiumTerrainProviderHeightmaps = new Cesium.CesiumTerrainProvider({
        url : 'https://assets.ion.cesium.com/us-east-1/asset_depot/1/CesiumWorldTerrain/v1.2',
        requestWaterMask: true,
        requestVertexNormals: true,
        proxy : new Cesium.DefaultProxy('/proxy/')
    });

    viewer.terrainProvider = cesiumTerrainProviderHeightmaps;
    */

    function loadTexture(figureFolder, view) {
        return new Promise(function(resolve) {
            Cesium.loadImage(figureFolder + '/body_parts_' + view + '_texture.png').then(resolve);
        })
    }

    function combineTextures(images) {
        const canvas = document.createElement('canvas');
        canvas.width = body_size * images.length;
        canvas.height = body_size;
        const context = canvas.getContext('2d');

        for (let i = 0; i < images.length; i++) {
            const image = images[i];
            context.drawImage(image, body_size * i, 0, body_size, body_size);
        }

        const texture = new Texture({
            context: scene.context,
            source: canvas
        });

        return texture;
    }

    async function loadSkeleton(figureFolder) {
        const response = await fetch(figureFolder + '/skeleton_front.json');
        const skeleton = await response.json();

        return skeleton;
    }

    async function loadSkeletonAnimation(figureFolder, transformation) {
        const response = await fetch(figureFolder + '/' + transformation + '_animation.npy');

        if (!response.ok)
            return Promise.resolve(undefined);

        return new Promise(function(resolve) {
            NumpyLoader.ajax(figureFolder + '/' + transformation + '_animation.npy', function (npy) {
                resolve(npy.data);
            });
        });
    }

    function createAnimationSkeleton(skeleton_animation, frame) {
        const skeleton = {};

        for (let index = 0; index < 16; index++) {
            const x = skeleton_animation[frame * (16 * 3) + index * 3];
            const y = skeleton_animation[frame * (16 * 3) + index * 3 + 1];
            const z = skeleton_animation[frame * (16 * 3) + index * 3 + 2];
            skeleton[String(index)] = [x, y, z];
        }

        return skeleton;
    }

    function transformSkeleton(skeleton) {
        const skeletonPoints = [];

        for (let point of Object.values(skeleton)) {
            point = point.slice();
            point[1] = 600 - point[1];
            point = point.map(coord => (coord - 300) / (600 / body_size))
            skeletonPoints.push(point);
        }

        return skeletonPoints;
    }

    function loadBodyPart(figureFolder, bone_name) {
        return new Promise(function(resolve, reject) {
            NumpyLoader.ajax(figureFolder + '/' + bone_name + '.npy', function (npy) {
                const texture = new Texture({
                    context: scene.context,
                    pixelDatatype: PixelDatatype.FLOAT,
                    pixelFormat: PixelFormat.ALPHA,
                    source: {
                        width: 512,
                        height: 512,
                        arrayBufferView: npy.data
                    },
                    sampler: new Sampler({
                        minificationFilter: TextureMagnificationFilter.NEAREST,
                        magnificationFilter: TextureMagnificationFilter.NEAREST
                    })
                });

                resolve(texture);
            });
        });
    }

    function calculateMidpoint(points) {
        const numPoints = points.length;
        const midpoint = Array.from({ length: 3 }, () => 0);

        for (const point of points) {
            for (let i = 0; i < 3; i++) {
                midpoint[i] += point[i] / numPoints;
            }
        }

        return midpoint;
    }

    function calculateBodyPartMidpoint(bone_name, keypoints, skeleton) {
        const pose_points = [];

        for (const keypoint of keypoints) {
            const keypoint_index = POSE_POINTS[keypoint];
            const pose_point = skeleton[String(keypoint_index)];
            pose_points.push(pose_point);
        }

        let mid_point = calculateMidpoint(pose_points);
        mid_point[1] = 600 - mid_point[1];
        mid_point = mid_point.map(coord => (coord - 300) / (600 / body_size));

        return mid_point;
    }

    async function loadScale(figureFolder, bone_name) {
        const response = await fetch(figureFolder + '/' + bone_name + '.json');
        const body_part_metadata = await response.json();

        const scale = body_part_metadata["scale"] * scaling_factor;
        return scale;
    }

    /*
    const div = document.createElement('div');
    div.style.position = 'absolute';
    div.style.top = '10px';
    div.style.left = '10px';
    div.style.color = 'white';
    div.style['font-size'] = '64px';
    div.style['font-family'] = 'sans-serif';
    div.innerText = "Festivals in Switzerland";

    document.body.appendChild(div);


    const speechBubbleLayer = new SpeechBubbleLayer(viewer);
    let speechBubble = speechBubbleLayer.add("To which other festivals have you been this year?",
        Cesium.Cartesian3.fromDegrees(8.54087402600246 + 0.0005, 47.367151629617496 + 0.0005, 70)
        );

    document.addEventListener("keydown", (event) => {
        speechBubbleLayer.remove(speechBubble);

        speechBubble = speechBubbleLayer.add("I have been to the OpenAir St. Gallen in June.",
            Cesium.Cartesian3.fromDegrees(8.54087402600246 - 0.0005, 47.367151629617496 + 0.0005, 70)
        );

        setTimeout(() => {
            speechBubbleLayer.remove(speechBubble);

            speechBubble = speechBubbleLayer.add("What about you?",
                Cesium.Cartesian3.fromDegrees(8.54087402600246 - 0.0005, 47.367151629617496 + 0.0005, 70)
            );

            setTimeout(() => {
                speechBubbleLayer.remove(speechBubble);

                speechBubble = speechBubbleLayer.add("I just visited the Relevatez Festival near Lucerne.",
                    Cesium.Cartesian3.fromDegrees(8.54087402600246 + 0.0005, 47.367151629617496 + 0.0005, 70)
                );

                setTimeout(() => {
                    speechBubbleLayer.remove(speechBubble);
                }, 4000);
            }, 4000);
        }, 4000);
    });
    */

    function combineDataToTexture(dataArrays) {
        let combinedArrays = [];

        for (const dataArray of dataArrays) {
            const numElements = dataArray.length;
            const missingElements = Array.from({ length: 64 - numElements }, () => 0);
            const filledArray = dataArray.concat(missingElements);
            combinedArrays = combinedArrays.concat(filledArray);
        }

        const combinedData = new Float32Array(combinedArrays);

        const texture = new Texture({
            context: scene.context,
            pixelDatatype: PixelDatatype.FLOAT,
            pixelFormat: PixelFormat.ALPHA,
            flipY: false,
            source: {
                width: 64,
                height: dataArrays.length,
                arrayBufferView: combinedData
            },
            sampler: new Sampler({
                minificationFilter: TextureMagnificationFilter.NEAREST,
                magnificationFilter: TextureMagnificationFilter.NEAREST
            })
        });

        return texture;
    }

    async function init(figureFolder, shift, modelRotation, headbangRotation) {
        const skeleton = await loadSkeleton(figureFolder);
        let skeleton_animation, rotation_animation, num_frames;

        try {
            skeleton_animation = await loadSkeletonAnimation(figureFolder, "skeleton");
            rotation_animation = await loadSkeletonAnimation(figureFolder, "rotation");
            num_frames = skeleton_animation.length / 16 / 3;
        } catch(e) {
        }

        const skeleton_points = transformSkeleton(skeleton);
        const bodyPartPromises = [];
        const scalePromises = [];
        const bodyPartMidpoints = [];
        const bodyPartTranslations = [];
        const bodyPartRotations = [];

        if (skeleton_animation !== undefined) {
            for (let i = 0; i < num_frames; i++) {
                bodyPartTranslations[i] = Array.from({ length: 14 * 3 }, () => 0);
                bodyPartRotations[i] = Array.from({ length: 14 * 4 }, () => 0);

                for (let j = 0; j < 14 * 4; j++) {
                    bodyPartRotations[i][j] = rotation_animation[i * (14 * 4) + j];
                }
            }
        }

        let boneIndex = 0;

        for (const [bodyPartName, points] of Object.entries(BONES)) {
            const bodyPartPromise = loadBodyPart(figureFolder, bodyPartName, points, skeleton);
            bodyPartPromises.push(bodyPartPromise);

            const boneMidpoint = calculateBodyPartMidpoint(bodyPartName, points, skeleton);
            bodyPartMidpoints.push(boneMidpoint);

            if (skeleton_animation !== undefined) {
                for (let i = 0; i < num_frames; i++) {
                    const animationSkeleton = createAnimationSkeleton(skeleton_animation, i);
                    const boneMidpointAnimation = calculateBodyPartMidpoint(bodyPartName, points, animationSkeleton);
                    bodyPartTranslations[i][boneIndex * 3] = boneMidpointAnimation[0] - boneMidpoint[0];
                    bodyPartTranslations[i][boneIndex * 3 + 1] = boneMidpointAnimation[1] - boneMidpoint[1];
                    bodyPartTranslations[i][boneIndex * 3 + 2] = boneMidpointAnimation[2] - boneMidpoint[2];
                }
            }

            const scalePromise= loadScale(figureFolder, bodyPartName);
            scalePromises.push(scalePromise);

            boneIndex++;
        }

        const body_parts = await Promise.all(bodyPartPromises);
        const scales = await Promise.all(scalePromises);

        const texturePromises = VIEWS.map(view => loadTexture(figureFolder, view));
        const images = await Promise.all(texturePromises);
        const texture = combineTextures(images);

        let transformationDataFunction;
        let timer = 0;
        const randomness = false;

        if (skeleton_animation !== undefined) {
            transformationDataFunction = function() {
                timer = (timer + 1);

                const transformationData = combineDataToTexture([
                    skeleton_points.flat(),
                    bodyPartMidpoints.flat(),
                    bodyPartTranslations[timer],
                    bodyPartRotations[timer]
                ]);

                return transformationData;
            };
        } else if (randomness) {
            transformationDataFunction = function() {
                const translation = Array.from({ length: 14 * 3 }, () => 0);
                const rotation = Array.from({ length: 14 * 4 }, () => Math.random() * 0.2 - 0.1);

                const transformationData = combineDataToTexture([
                    skeleton_points.flat(),
                    bodyPartMidpoints.flat(),
                    translation,
                    rotation
                ]);

                return transformationData;
            };
        } else {
            const transformationData = combineDataToTexture([
                skeleton_points.flat(),
                bodyPartMidpoints.flat(),
                bodyPartTranslations,
                bodyPartRotations
            ]);

            transformationDataFunction = function() {
                return transformationData;
            };
        }

        //const positionProperty = await interpolatePositions(Cesium, stopTime, startTime);

        for (let i = 0; i < 1; i++) {
            const billboards = scene.primitives.add(new Cesium.BillboardCollection());

            const billboard = billboards.add({
                position : Cesium.Cartesian3.fromDegrees(8.54087402600246 + shift.lon, 47.367151629617496 + shift.lat, shift.height), //positionProperty,
                //scaleByDistance: new Cesium.NearFarScalar(100, 10, 5000000, 0.5),
                modelRotation: modelRotation,
                headbangRotation: headbangRotation,
                image: './Sandcastle/images/whiteShapes.png',
                bodyParts: body_parts,
                transformationDataFunction: transformationDataFunction,
                bodyPartScales: scales,
                adjustedBodySize: adjusted_body_size,
                imageTexture: texture,
                width: 256,
                height: 256,
            });

            /*window.setInterval(function() {
                const position = positionProperty.getValue(viewer.clock.currentTime);
                console.log(position);
                billboard.position = position;
            }, 1000);*/
        }


        /*const tracks = await Cesium.GeoJsonDataSource.load(
            "geodata/pilgerwege.geojson",
            {
                stroke: Cesium.Color.fromCssColorString("#ffffb3"),
                clampToGround: true,
                strokeWidth: 10
            }
        );

        viewer.dataSources.add(tracks);
        viewer.zoomTo(tracks); */
    }

    /*
    var hpr = new Cesium.HeadingPitchRoll(Math.PI, 0, 0);
    var origin = Cesium.Cartesian3.fromDegrees(8.54087402600246, 47.367151629617496, 10);

    var entity = scene.primitives.add(Cesium.Model.fromGltf({
        url : "geodata/stage.gltf",
        scale: 1,
        modelMatrix : Cesium.Transforms.headingPitchRollToFixedFrame(origin, hpr)
    }));
     */

    //init("figures/68a6d698-d32c-4701-8826-72239b618960_rw_3840", {lat: -0.00025, lon: 0, height: 35}, {x: -Math.PI/4, y: -Math.PI/2, z: 0}, {x: 0, y: 0, z: 1});
    //init("figures/braun_hogenberg_iii_4_b1", {lat: 0.0005, lon: -0.0005, height: 30}, {x: -Math.PI/4, y: 0, z: 0}, {x: 1, y: 0, z: -1});
    init("figures/Patagonischer_Riese_links", {lat: 0.0005, lon: 0.0005, height: 30}, {x: -Math.PI/4, y: 0, z: 0}, {x: -1, y: 0, z: 0});
});
