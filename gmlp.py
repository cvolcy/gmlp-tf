import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.activations import gelu

class gMLPLayer(Model):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(gMLPLayer, self).__init__(**kwargs)

        self.norm = BatchNormalization()
        self.proj_in = Dense(dim_in, activation="gelu")
        self.sgu = SpatialGatingUnit()
        self.proj_out = Dense(dim_out, activation="gelu")

    def call(self, inputs):
        shortcut = inputs
        x = self.norm(inputs)
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x + shortcut

class SpatialGatingUnit(Layer):
    def __init__(self, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.norm = LayerNormalization()
        
    def call(self, inputs):
        u, v = tf.split(inputs, 2, axis=-1)
        v = self.norm(v)

        return u * v