from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Add, Layer, BatchNormalization, Activation
from config import BATCH_SIZE


class Conv2DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = 'relu', *args, **kwargs):
        super(Conv2DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (BATCH_SIZE > BATCH_SIZE):
            x = self.bn(x) 
        x = self.activation(x)
               
        return x


class Conv2DTransposeBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = 'relu', *args, **kwargs):
        super(Conv2DTransposeBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (BATCH_SIZE > BATCH_SIZE):
            x = self.bn(x) 
        x = self.activation(x)
        
        return x


def down(num_filters, layer):
    pool = Conv2DBatch(num_filters * 2, 3, strides = (2,2))(layer)
    pool = Conv2DBatch(num_filters * 2, 3)(pool)
    pool = Conv2DBatch(num_filters * 2, 3)(pool)
    
    return pool


def up(num_filters, layer, other_layer):
    deconv = Conv2DTransposeBatch(num_filters, 3, strides = (2,2))(layer)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    deconv = Add()([deconv,other_layer])
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    
    return deconv


def create_model(image_size, num_body_parts, num_filters):
	depth_image = Input((image_size, image_size, num_body_parts))

	conv1 = Conv2D(num_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(depth_image)
	conv2 = down(num_filters, conv1)
	conv3 = down(num_filters * 2, conv2)
	conv4 = down(num_filters * 4, conv3)
	conv5 = down(num_filters * 8, conv4)

	up4 = up(num_filters * 8, conv5, conv4)
	up3 = up(num_filters * 4, up4, conv3)
	up2 = up(num_filters * 2, up3, conv2)
	up1 = up(num_filters * 1, up2, conv1)

	uv_image = Conv2D(3, 1, activation = 'sigmoid')(up1)

	model = Model(inputs=depth_image, outputs=uv_image)
	model.summary()
	
	return model