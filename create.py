import numpy as np
#import matplotlib.pyplot as plt

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle

from create_networks import build_mlp, build_conv, build_vgg_16, build_conv2D, build_conv2DLarger, build_conv2DLargerPool, build_conv2DSite
from augment_data import augment

def load_data(nr, input_dim,type):
    if input_dim == 50:
        train = pickle.load( open( "traj_50/train"+str(nr)+"_50_traj.p", "rb" ) )
    elif input_dim == 100:
        train = pickle.load( open( "traj_100/train"+str(nr)+"_100_traj.p", "rb" ) )
    elif input_dim == 16:
        train = pickle.load( open( "traj_16/train"+str(nr)+"_16_traj.p", "rb" ) )
    else:
        train = pickle.load( open( "traj_sub_20/train"+str(nr)+"_traj.p", "rb" ) )
    train_X = []
    train_y = []

    mapping = {}
    id = 0
    for appliances in train.values():
        amount_samples = len(appliances)
        for i in range(amount_samples):
            if type == 'vgg' or type == 'conv2D' or type == 'conv2DLarger' or type == 'conv2DLargerPool' or type == 'build_conv2DSite':
                train_X.append(np.reshape(appliances[i],(1,len(appliances[i]),-1)))
            else:
                train_X.append(appliances[i])
            train_y.append(id)
        mapping[train.keys()[id]] = id
        id += 1

    if input_dim == 50:
        test = pickle.load( open( "traj_50/test"+str(nr)+"_50_traj.p", "rb" ) )
    elif input_dim == 100:
        test = pickle.load( open( "traj_100/test"+str(nr)+"_100_traj.p", "rb" ) )
    elif input_dim == 16:
        test = pickle.load( open( "traj_16/test"+str(nr)+"_16_traj.p", "rb" ) )
    else:
        test = pickle.load( open( "traj_sub_20/test"+str(nr)+"_traj.p", "rb" ) )
    test_X = []
    test_y = []

    id = 0
    for appliances in test.values():
        amount_samples = len(appliances)
        for i in range(amount_samples):
            if type == 'vgg' or type == 'conv2D' or type == 'conv2DLarger' or type == 'conv2DLargerPool' or type == 'build_conv2DSite':
                test_X.append(np.reshape(appliances[i],(1,len(appliances[i]),-1)))
            else:
                test_X.append(appliances[i])
            test_y.append(mapping[test.keys()[id]])
        id += 1

    return np.array(train_X, dtype = 'float32'), np.array(train_y, dtype = 'int32'), np.array(test_X, dtype = 'float32'), np.array(test_y, dtype = 'int32')

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def createNetwork(input_dim, type, number_of_filters):
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')

    if type == 'mlp':
        network = build_mlp(input_dim,input_var)
    elif type == 'conv':
        network = build_conv(input_dim,input_var)
    elif type == 'conv2D':
        input_var = T.tensor4('inputs')
        network = build_conv2D(input_dim,input_var)
    elif type == 'conv2DLarger':
        input_var = T.tensor4('inputs')
        network = build_conv2DLarger(input_dim,number_of_filters,input_var)
    elif type == 'conv2DLargerPool':
        input_var = T.tensor4('inputs')
        network = build_conv2DLargerPool(input_dim,number_of_filters,input_var)
    elif type == 'build_conv2DSite':
        input_var = T.tensor4('inputs')
        network = build_conv2DLargerPool(input_dim,number_of_filters,input_var)
    elif type == 'vgg':
        input_var = T.tensor4('inputs')
        network = build_vgg_16(input_dim, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    test_equal = T.sum(T.eq(T.argmax(test_prediction, axis=1), target_var))
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_equal])

    test_function = theano.function([input_var], prediction)

    return train_fn, val_fn, test_function

def training(iterations, tr_error, input_dim, network_type, aug, number_of_filters):
    sum_acc = 0
    file_name = "{}_{}_{}_{}_{}_{}_fixed.txt".format(network_type, iterations, tr_error,input_dim,aug, number_of_filters)

    confusion_matrix = np.zeros((11,11))
    sum_correct = 0
    sum_total = 0
    for nr in range(1,56):
        train_fn, val_fn, test_function = createNetwork(input_dim, network_type, number_of_filters)

        train_X, train_y, test_X, test_y = load_data(nr, input_dim, network_type)

        if aug:
            train_X, train_y = augment(train_X, train_y, aug, network_type)


        train_err = 0
        it = 0
        acc = 0
        #while acc < tr_error and it < iterations:
        while it < iterations:
            for batch in iterate_minibatches(train_X, train_y, 32):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)

            err, acc, correct = val_fn(train_X, train_y)
            it += 1
        print('Training: loss {}, acc {}'.format(err, acc))

        f = open( file_name, "a" )
        f.write('Training: loss {}, acc {} \n'.format(err, acc))

        loss, acc, correct = val_fn(test_X, test_y)
        sum_acc += acc

        ypred = test_function(test_X)
        for yarray, ind2 in zip(ypred,test_y):
            ind1 = np.argmax(yarray)
            confusion_matrix[ind1,ind2] +=1
            confusion_matrix_house[ind1,ind2] += 1
            sum_total += 1.0
            if ind1 == ind2:
                sum_correct += 1.0

        print('Testing: loss {}, acc {}, iterations {}, correct {}'.format(loss, acc, it, correct))
        f.write('Testing: loss {}, acc {}, iterations {},  correct {} \n'.format(loss, acc, it, correct))
        f.close()

        print(confusion_matrix)


    print('Average accuracy: {}'.format(sum_acc / 55))
    print('Sum correct: {} / {} = {}'.format(sum_correct,sum_total,sum_correct/sum_total))

    f = open( file_name, "a" )
    f.write('Average accuracy: {} \n'.format(sum_acc / 55))
    f.write('Sum correct: {} \n'.format(sum_correct))
    f.write( 'Confussion matrix: {} \n'.format(confusion_matrix) )
    f.write( 'Confussion matrix: {} \n'.format(confusion_matrix / sum(sum(confusion_matrix))) )
    f.close()

if __name__ == "__main__":
    if False:
        # network_type = mlp, conv, vgg
        # augm -> factor waarmee je wilt augmenten
        iterations = [100] #100
        errors = [0.9999] #0.9999
        size = [16]
        augment_factors = [50] #50

        # network_type, iterations, tr_error,input_dim,aug
        # (iterations, tr_error, input_dim, network_type, aug)
        for it in iterations:
            for e in errors:
                for s in size:
                    for a in augment_factors:
                        print("Iterations {}, train error {}, size {}, augmentation {} ".format(it,e,s,a))
                        training(it, e, s, 'mlp', a)

    else:
        print('Conv')
        for s in [100]:
            for number_of_filters in [100]:
                training(100,0.99,s,'build_conv2DSite',0, number_of_filters)
