import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, LayerNormalization
from tensorflow.keras.initializers import RandomUniform

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
    def __init__(self, dim_in, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        
        self.dim_in = dim_in

    def build(self, input_shape):
        self.norm = LayerNormalization()
        
        self.conv1d_bias = self.add_weight(
            name="sgu_conv1d_bias",
            shape=[self.dim_in], 
            initializer=tf.ones
        )
        
        init_range = (1e-3) / self.dim_in
        self.conv1d_filter = self.add_weight(
            name="sgu_conv1d_filter", 
            shape=(1, self.dim_in, self.dim_in), 
            initializer=RandomUniform(minval=-init_range, maxval=init_range)
        )
        
    def call(self, inputs):
        u, v = tf.split(inputs, 2, axis=-1)
        v = self.norm(v)

        v = tf.transpose(v, (0, 2, 1))
        v = tf.nn.conv1d(v, filters=self.conv1d_filter, stride=1, use_cudnn_on_gpu=True, data_format="NWC", padding="VALID") 
        v = tf.nn.bias_add(v, bias=self.conv1d_bias, data_format="NWC")
        v = tf.transpose(v, (0,2,1))

        return u * v