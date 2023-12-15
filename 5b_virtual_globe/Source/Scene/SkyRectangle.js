/*global define*/
define([
        '../Core/BoxGeometry',
        '../Core/RectangleGeometry',
        '../Core/Rectangle',
        '../Core/Cartesian3',
        '../Core/defaultValue',
        '../Core/defined',
        '../Core/destroyObject',
        '../Core/DeveloperError',
        '../Core/GeometryPipeline',
        '../Core/Matrix4',
        '../Core/VertexFormat',
        '../Core/PrimitiveType',
        '../Renderer/Buffer',
        '../Renderer/BufferUsage',
        '../Renderer/CubeMap',
        '../Renderer/DrawCommand',
        '../Renderer/loadCubeMap',
        '../Renderer/RenderState',
        '../Renderer/ShaderProgram',
        '../Renderer/ShaderSource',
        '../Renderer/VertexArray',
        '../Shaders/SkyRectangleFS',
        '../Shaders/SkyRectangleVS',
        './BlendingState',
        './SceneMode'
    ], function(
        BoxGeometry,
        RectangleGeometry,
        Rectangle,
        Cartesian3,
        defaultValue,
        defined,
        destroyObject,
        DeveloperError,
        GeometryPipeline,
        Matrix4,
        VertexFormat,
        PrimitiveType,
        Buffer,
        BufferUsage,
        CubeMap,
        DrawCommand,
        loadCubeMap,
        RenderState,
        ShaderProgram,
        ShaderSource,
        VertexArray,
        SkyRectangleFS,
        SkyRectangleVS,
        BlendingState,
        SceneMode) {
    'use strict';

    /**
     * A sky box around the scene to draw stars.  The sky box is defined using the True Equator Mean Equinox (TEME) axes.
     * <p>
     * This is only supported in 3D.  The sky box is faded out when morphing to 2D or Columbus view.  The size of
     * the sky box must not exceed {@link Scene#maximumCubeMapSize}.
     * </p>
     *
     * @alias SkyBox
     * @constructor
     *
     * @param {Object} options Object with the following properties:
     * @param {Object} [options.sources] The source URL or <code>Image</code> object for each of the six cube map faces.  See the example below.
     * @param {Boolean} [options.show=true] Determines if this primitive will be shown.
     *
     *
     * @example
     * scene.skyBox = new Cesium.SkyBox({
     *   sources : {
     *     positiveX : 'skybox_px.png',
     *     negativeX : 'skybox_nx.png',
     *     positiveY : 'skybox_py.png',
     *     negativeY : 'skybox_ny.png',
     *     positiveZ : 'skybox_pz.png',
     *     negativeZ : 'skybox_nz.png'
     *   }
     * });
     *
     * @see Scene#skyBox
     * @see Transforms.computeTemeToPseudoFixedMatrix
     */
    function SkyRectangle(camera) {
        /**
         * Determines if the sky box will be shown.
         *
         * @type {Boolean}
         * @default true
         */
        this.show = true;
        this.camera = camera;

        this._command = new DrawCommand({
            modelMatrix : Matrix4.clone(Matrix4.IDENTITY),
            primitiveType: PrimitiveType.TRIANGLE_STRIP,
            owner : this
        });
    }

    /**
     * Called when {@link Viewer} or {@link CesiumWidget} render the scene to
     * get the draw commands needed to render this primitive.
     * <p>
     * Do not call this function directly.  This is documented just to
     * list the exceptions that may be propagated when the scene is rendered:
     * </p>
     *
     * @exception {DeveloperError} this.sources is required and must have positiveX, negativeX, positiveY, negativeY, positiveZ, and negativeZ properties.
     * @exception {DeveloperError} this.sources properties must all be the same type.
     */
    SkyRectangle.prototype.update = function(frameState) {
        return undefined;

        if (!this.show) {
            return undefined;
        }

        if ((frameState.mode !== SceneMode.SCENE3D) &&
            (frameState.mode !== SceneMode.MORPHING)) {
            return undefined;
        }

        var context = frameState.context;
        var camera = this.camera;

        var command = this._command;

        if (!defined(command.vertexArray)) {
            var vertexBuffer = Buffer.createVertexBuffer({
                context : context,
                typedArray : new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]),
                usage : BufferUsage.STATIC_DRAW
            });

            command.vertexArray = new VertexArray({
                context : context,
                attributes: [{
                    vertexBuffer: vertexBuffer,
                    componentsPerAttribute: 2,
                    attributeLocations : {
                        position : 0
                    },
                    bufferUsage : BufferUsage.STATIC_DRAW
                }]
            });

            //var pickFS = ShaderSource.createPickFragmentShaderSource(SkyRectangleFS, 'uniform');

            command.shaderProgram = ShaderProgram.fromCache({
                context : context,
                vertexShaderSource : SkyRectangleVS,
                fragmentShaderSource : SkyRectangleFS,
                attributeLocations : {
                    position : 0
                }
            });

            command.renderState = RenderState.fromCache({
                blending : BlendingState.ALPHA_BLEND,
                depthTest : {
                    enabled : true
                }
            });

            command.uniformMap = {
                cameraForward: function() {
                    return camera.direction;
                },
                cameraUp: function() {
                    return camera.up;
                },
                cameraRight: function() {
                    return camera.right;
                },
                tanCameraFovHalf: function() {
                    return Math.tan(camera.frustum.fov / 2);
                },
                tanCameraFovYHalf: function() {
                    return Math.tan(camera.frustum.fovy / 2);
                },
                depthA: function() {
                    var near = context.uniformState.currentFrustum.x;
                    var far = context.uniformState.currentFrustum.y;
                    return (far+near)/(far-near);
                },
                depthB: function() {
                    var near = context.uniformState.currentFrustum.x;
                    var far = context.uniformState.currentFrustum.y;
                    return 2.0*far*near/(far-near);
                }
            };
        }

        return command;
    };

    /**
     * Returns true if this object was destroyed; otherwise, false.
     * <br /><br />
     * If this object was destroyed, it should not be used; calling any function other than
     * <code>isDestroyed</code> will result in a {@link DeveloperError} exception.
     *
     * @returns {Boolean} <code>true</code> if this object was destroyed; otherwise, <code>false</code>.
     *
     * @see SkyBox#destroy
     */
    SkyRectangle.prototype.isDestroyed = function() {
        return false;
    };

    /**
     * Destroys the WebGL resources held by this object.  Destroying an object allows for deterministic
     * release of WebGL resources, instead of relying on the garbage collector to destroy this object.
     * <br /><br />
     * Once an object is destroyed, it should not be used; calling any function other than
     * <code>isDestroyed</code> will result in a {@link DeveloperError} exception.  Therefore,
     * assign the return value (<code>undefined</code>) to the object as done in the example.
     *
     * @returns {undefined}
     *
     * @exception {DeveloperError} This object was destroyed, i.e., destroy() was called.
     *
     *
     * @example
     * skyBox = skyBox && skyBox.destroy();
     *
     * @see SkyBox#isDestroyed
     */
    SkyRectangle.prototype.destroy = function() {
        var command = this._command;
        command.vertexArray = command.vertexArray && command.vertexArray.destroy();
        command.shaderProgram = command.shaderProgram && command.shaderProgram.destroy();
        return destroyObject(this);
    };

    return SkyRectangle;
});
