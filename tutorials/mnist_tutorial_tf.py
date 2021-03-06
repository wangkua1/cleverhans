from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
K = backend
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, _FlagsWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import cnn_model, fc_model, AccuracyReport
import cPickle as pkl
import math
import numpy as np

FLAGS = flags.FLAGS



def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.1, nb_filters=64, dropout=0,
                   model_name='cnn'):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # keras.layers.core.K.set_learning_phase(1)
    # backend.set_learning_phase(1)
    print ('...')
    # backend.set_learning_phase(tf.placeholder(dtype='bool',name='custome_ph'))
    # _GRAPH_LEARNING_PHASES[tf.get_default_graph()] = tf.placeholder(dtype='bool',name='custome_ph')
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print (backend.learning_phase())

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # K._LEARNING_PHASE = tf.constant(0)
    # Define TF model graph
    model = eval(model_name+'_model')(nb_filters=nb_filters, dropout=dropout)
    # model = fc_model(nb_filters=nb_filters, dropout=dropout)
    preds = model(x)
    print("Defined TensorFlow model graph.")

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                args=train_params)
    # keras.layers.core.K.set_learning_phase(0)
    # backend.set_learning_phase(0)
    # print (backend.learning_phase())
    ## get y's
    # ensembled y's
    ys = get_y(sess, x, y, preds, X_test, Y_test, learning_phase=0,args={'batch_size': batch_size})
    pkl.dump(ys, open('{}_ens_ys.p'.format(model_name),'wb'))
    # sampled y's
    T = 100
    allys = []
    for idx in xrange(T):
        ys = get_y(sess, x, y, preds, X_test, Y_test, learning_phase=1,args={'batch_size': batch_size})
        # ys = model.predict(X_test)
        allys.append(ys)
    # print (backend.learning_phase())
    pkl.dump(allys, open('{}_sampled_ys.p'.format(model_name),'wb'))
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'eps': 0.3}
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc
    ## get y's
    # ensembled y's
    ys = get_y(sess, x, y, preds_adv, X_test, Y_test, learning_phase=0,args={'batch_size': batch_size})
    pkl.dump(ys, open('{}_adv_ens_ys.p'.format(model_name),'wb'))
    # sampled y's
    T = 100
    allys = []
    for idx in xrange(T):
        ys = get_y(sess, x, y, preds_adv, X_test, Y_test, learning_phase=1,args={'batch_size': batch_size})
        # ys = model.predict(X_test)
        allys.append(ys)
    # print (backend.learning_phase())
    pkl.dump(allys, open('{}_adv_sampled_ys.p'.format(model_name),'wb'))
    #.....................................
    # print("Repeating the process, using adversarial training")
    # # Redefine TF model graph
    # model_2 = cnn_model()
    # preds_2 = model_2(x)
    # fgsm2 = FastGradientMethod(model_2, sess=sess)
    # preds_2_adv = model_2(fgsm2.generate(x, **fgsm_params))

    # def evaluate_2():
    #     # Accuracy of adversarially trained model on legitimate test inputs
    #     eval_params = {'batch_size': batch_size}
    #     accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
    #                           args=eval_params)
    #     print('Test accuracy on legitimate examples: %0.4f' % accuracy)
    #     report.adv_train_clean_eval = accuracy

    #     # Accuracy of the adversarially trained model on adversarial examples
    #     accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
    #                           Y_test, args=eval_params)
    #     print('Test accuracy on adversarial examples: %0.4f' % accuracy)
    #     report.adv_train_adv_eval = accuracy

    # # Perform and evaluate adversarial training
    # model_train(sess, x, y, preds_2, X_train, Y_train,
    #             predictions_adv=preds_2_adv, evaluate=evaluate_2,
    #             args=train_params)

    return report
# def get_y(sess, x, y, model, X_test, Y_test, args=None):


def get_y(sess, x, y, model, X_test, Y_test, learning_phase = 0, args=None):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param model: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :return: a float with the accuracy value
    """
    ret_y = []
    args = _FlagsWrapper(args or {})
    assert args.batch_size, "Batch size was not given in args dict"
    with sess.as_default():
        # Compute number of batches
        nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
        assert nb_batches * args.batch_size >= len(X_test)

        for batch in range(nb_batches):
            if batch % 100 == 0 and batch > 0:
                print("Batch " + str(batch))

            # Must not use the `batch_indices` function here, because it
            # repeats some examples.
            # It's acceptable to repeat during training, but not eval.
            start = batch * args.batch_size
            end = min(len(X_test), start + args.batch_size)
            cur_batch_size = end - start

            # The last batch may be smaller than all others, so we need to
            # account for variable batch size here
            # print (type(x))
            # print (type(backend.learning_phase()))
            cur_y = model.eval(
                feed_dict={x: X_test[start:end],
                          backend.learning_phase(): learning_phase}, session=sess)
            ret_y.append(cur_y)
        assert end >= len(X_test)
    return np.concatenate(ret_y, 0)

def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate, nb_filters=FLAGS.nb_filters, 
                   dropout=FLAGS.dropout,model_name=FLAGS.model_name)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
    flags.DEFINE_integer('nb_filters', 64, 'number of filters (channel size) at each Conv Layer')
    flags.DEFINE_float('dropout', 0, 'dropout rate (0=no dropout)')
    flags.DEFINE_string('model_name', 'cnn', '"fc" or "cnn"')
    app.run()
