import theano
import theano.tensor as T

import lasagne
import lasagne.layers
import lasagne.layers.dnn

def build_mlp(input_dim, input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, input_dim, input_dim),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=input_dim*input_dim*2,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=input_dim*input_dim*2,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=11,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_conv(input_dim, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, input_dim, input_dim),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv1DLayer(l_in,
                                        num_filters = 5,
                                        filter_size = 5,
                                        stride = 2)

    l_conv2 = lasagne.layers.Conv1DLayer(l_conv1,
                                        num_filters = 5,
                                        filter_size = 5,
                                        stride = 2)

    l_fully = lasagne.layers.DenseLayer(l_conv2,
                                        num_units = 100,
                                        nonlinearity = lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(l_fully,
                                        num_units = 11,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

def build_conv2D(input_dim, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, input_dim, input_dim),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                                        num_filters = 5,
                                        filter_size = 5,
                                        stride = (2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1,
                                        num_filters = 5,
                                        filter_size = 5,
                                        stride = (2,2))

    l_fully = lasagne.layers.DenseLayer(l_conv2,
                                        num_units = 100,
                                        nonlinearity = lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(l_fully,
                                        num_units = 11,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

def build_conv2DLarger(input_dim, number_of_filters, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, input_dim, input_dim),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                                        num_filters = number_of_filters,
                                        filter_size = 5,
                                        stride = (2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1,
                                        num_filters = number_of_filters,
                                        filter_size = 5,
                                        stride = (2,2))

    l_fully = lasagne.layers.DenseLayer(l_conv2,
                                        num_units = input_dim*input_dim,
                                        nonlinearity = lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(l_fully,
                                        num_units = 11,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

def build_conv2DLargerPool(input_dim, number_of_filters, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, input_dim, input_dim),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                                        num_filters = number_of_filters,
                                        filter_size = 5,
                                        stride = (2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(l_conv1,
                                        num_filters = number_of_filters,
                                        filter_size = 5,
                                        stride = (2,2))

    l_fully = lasagne.layers.DenseLayer(l_conv2,
                                        num_units = input_dim*input_dim,
                                        nonlinearity = lasagne.nonlinearities.rectify)

    l_reshape = lasagne.layers.reshape(l_fully,(-1,1,input_dim,input_dim))

    l_pool = lasagne.layers.MaxPool2DLayer(l_reshape,
                                            pool_size = 3,
                                            stride = 2)

    l_out = lasagne.layers.DenseLayer(l_pool,
                                        num_units = 11,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

def build_conv2DSite(input_dim, number_of_filters, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, input_dim, input_dim),
                                     input_var=input_var)

    l_conv1 = lasagne.layers.Conv2DLayer(l_in,
                                        num_filters = number_of_filters,
                                        filter_size = (5,5),
                                        stride = (2,2))

    l_pool = lasagne.layers.MaxPool2DLayer(l_conv1,
                                            pool_size = (2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(l_pool,
                                        num_filters = number_of_filters,
                                        filter_size = (5,5))

    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2,
                                            pool_size = (2,2))

    l_fully = lasagne.layers.DenseLayer(l_pool2,
                                        num_units = input_dim*input_dim,
                                        nonlinearity = lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(l_pool,
                                        num_units = 11,
                                        nonlinearity = lasagne.nonlinearities.softmax)

    return l_out

def build_vgg_16(input_dim, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None,1,input_dim, input_dim),
                                    input_var=input_var)

    l_conv1 = lasagne.layers.dnn.Conv2DDNNLayer(l_in,
                                                num_filters=64,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv2 = lasagne.layers.dnn.Conv2DDNNLayer(l_in,
                                                num_filters=64,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_pool = lasagne.layers.Pool2DLayer(l_conv2,pool_size=2)

    l_conv3 = lasagne.layers.dnn.Conv2DDNNLayer(l_pool,
                                                num_filters=128,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv4 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv3,
                                                num_filters=128,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_pool2 = lasagne.layers.Pool2DLayer(l_conv4,pool_size=2)

    l_conv5 = lasagne.layers.dnn.Conv2DDNNLayer(l_pool2,
                                                num_filters=256,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv6 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv5,
                                                num_filters=256,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv7 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv6,
                                                num_filters=256,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_pool3 = lasagne.layers.Pool2DLayer(l_conv7,pool_size=2)

    l_conv8 = lasagne.layers.dnn.Conv2DDNNLayer(l_pool3,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv9 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv8,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv10 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv9,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_pool4 = lasagne.layers.Pool2DLayer(l_conv10,pool_size=2)

    l_conv11 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv10,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv12 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv11,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_conv13 = lasagne.layers.dnn.Conv2DDNNLayer(l_conv12,
                                                num_filters=512,
                                                filter_size=3,
                                                pad=1,
                                                flip_filters=False)

    l_pool5 = lasagne.layers.Pool2DLayer(l_conv13,pool_size=2)

    l_fc = lasagne.layers.DenseLayer(l_pool5,num_units=4096)

    l_drop_out = lasagne.layers.DropoutLayer(l_fc, p = 0.5)

    l_fc2 = lasagne.layers.DenseLayer(l_drop_out,num_units=4096)

    l_drop_out2 = lasagne.layers.DropoutLayer(l_fc2,p=0.5)

    l_fc3 = lasagne.layers.DenseLayer(l_drop_out2,num_units=11,nonlinearity=None)

    l_out = lasagne.layers.NonlinearityLayer(l_fc3,nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
