from tensorflow.keras.layers import Layer, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization, Activation
from config import BATCH_SIZE
import tensorflow_addons as tfa 


class Conv1DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=1, activation = 'relu', *args, **kwargs):
        super(Conv1DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv1D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        if (BATCH_SIZE > 1):
            x = self.bn(x)   
        x = self.activation(x)
     
        return x
    
    
class Conv2DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = 'relu', *args, **kwargs):
        super(Conv2DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (BATCH_SIZE > 1):
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
        if (BATCH_SIZE > 1):
            x = self.bn(x) 
        x = self.activation(x)
        
        return x
    

# retrieves and interpolates values within a discrete grid
class InterpolateBilinear(Layer):
    
    def call(self, grid, query_points):
        return tfa.image.interpolate_bilinear(grid, query_points)