import argparse
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_neural_networks import utils, networks, mnist, example_networks
from my_neural_networks.minibatcher import MiniBatcher


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Training for MNIST')
    parser.add_argument('data_folder', metavar='DATA_FOLDER',
                        help='the folder that contains all the input data')

    parser.add_argument('-e', '--max_epochs', type=int, default=500,
                        help='max number of epochs for training each model. (DEFAULT: 500)')
    parser.add_argument('-s', '--shape', type=int, default=0,
                        help='Neural Network shape. 0 when shape is (784, 300, 100, 10), 1 when (784, 10)')
    parser.add_argument('-n', '--max_n_examples', type=int, default=20000,
                        help='max number of examples for training. (DEFAULT: 20000)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for gradient descent. (DEFAULT: 1e-4)')
    parser.add_argument('-i', '--impl', choices=['torch.nn', 'torch.autograd', 'my'], default='my',
                        help='choose the network implementation (DEFAULT: my)')

    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id to use. -1 means cpu (DEFAULT: -1)')
    parser.add_argument('-m', '--minibatch_size', type=int, default=-1,
                        help='minibatch_size. -1 means all. (DEFAULT: -1)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def one_hot(y, n_classes):
    """Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot


def save_learning_curve(train_accs, test_accs, train_sizes):
    """Plot a learning curve

        Plot 'training set sizes vs. accuracies'
    """
    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_accs, '--', linewidth=2, label='train')
    ax.plot(train_sizes, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('learning_curve.png')


def create_model(shape):
    logging.info('selec implementation: {}'.format(args.impl))
    if args.impl == 'torch.nn':
        # torch.nn implementation
        model = example_networks.TorchNeuralNetwork(shape, gpu_id=args.gpu_id)
    elif args.impl == 'torch.autograd':
        # torch.autograd implementation
        model = networks.AutogradNeuralNetwork(shape, gpu_id=args.gpu_id)
    else:
        # our implementation
        model = networks.BasicNeuralNetwork(shape, gpu_id=args.gpu_id)
    return model


def main():
    # DEBUG: fix seed
    # torch.manual_seed(29)

    # load data
    X_train, y_train = mnist.load_train_data(args.data_folder, max_n_examples=args.max_n_examples)
    X_test, y_test = mnist.load_test_data(args.data_folder)

    # reshape the images into one dimension
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train_1hot = one_hot(y_train, mnist.N_CLASSES)
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test_1hot = one_hot(y_test, mnist.N_CLASSES)

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)

    # get network shape
    if args.shape == 0:
        shape = [X_train.shape[1], 300, 100, mnist.N_CLASSES]
    elif args.shape == 1:
        shape = [X_train.shape[1], mnist.N_CLASSES]
    print(shape)
    # if we want to run it with torch.autograd, we need to use Variable
    if args.impl != 'my':
        X_train = torch.autograd.Variable(X_train, requires_grad=True)
        y_train = torch.autograd.Variable(y_train, requires_grad=False)
        y_train_1hot = torch.autograd.Variable(y_train_1hot, requires_grad=False)
        X_test = torch.autograd.Variable(X_test, requires_grad=True)
        y_test = torch.autograd.Variable(y_test, requires_grad=False)
        y_test_1hot = torch.autograd.Variable(y_test_1hot, requires_grad=False)
        
        n_examples = X_train.data.shape[0]
        logging.info("X_train shape: {}".format(X_train.data.shape))
        logging.info("X_test shape: {}".format(X_test.data.shape))
    else:
        n_examples = X_train.shape[0]
        logging.info("X_train shape: {}".format(X_train.shape))
        logging.info("X_test shape: {}".format(X_test.shape))

    # if gpu_id is specified
    if args.gpu_id != -1:
        # move all variables to cuda
        X_train = X_train.cuda(args.gpu_id)
        y_train = y_train.cuda(args.gpu_id)
        y_train_1hot = y_train_1hot.cuda(args.gpu_id)
        X_test = X_test.cuda(args.gpu_id)
        y_test = y_test.cuda(args.gpu_id)
        y_test_1hot = y_test_1hot.cuda(args.gpu_id)

    # start training
    train_accs = []
    test_accs = []
    train_sizes = []
    for train_size in torch.arange(250, n_examples, 250):
        train_size = int(train_size)
        logging.info("--------------- training set size = {} ---------------".format(train_size))

        # create a model
        model = create_model(shape)
        # prepare batcher
        batcher = MiniBatcher(args.minibatch_size, train_size) if args.minibatch_size > 0 \
                    else MiniBatcher(train_size, train_size)
        # train the model with an early stop stratege
        previous_train_acc = None
        previous_loss = None
        for i_epoch in range(args.max_epochs):
            logging.debug("== EPOCH {} ==".format(i_epoch))
            for train_idxs in batcher.get_one_batch():
                # numpy to torch
                #train_idxs = torch.LongTensor(train_idxs)
                if args.gpu_id != -1:
                    train_idxs = train_idxs.cuda(args.gpu_id)

                X_train_cur, y_train_cur, y_train_1hot_cur = \
                        X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs]

                # fit to the training data
                loss = model.train_one_epoch(X_train_cur, y_train_cur, y_train_1hot_cur, args.learning_rate)

            logging.debug("loss = {}".format(loss))
            
            # early stop checking
            if previous_loss is not None and abs(previous_loss - loss) < 1e-3:
                logging.info("Early stop at epoch {}".format(i_epoch))
                break
            else:
                if previous_loss is not None:
                    logging.debug("diff = {}".format(abs(previous_loss - loss)))
                previous_loss = loss
        
        # test the trained model
        y_train_pred = model.predict(X_train_cur)
        y_test_pred = model.predict(X_test)
        train_acc = utils.accuracy(y_train_cur, y_train_pred)
        test_acc = utils.accuracy(y_test, y_test_pred)
        logging.info("loss = {}".format(loss))
        logging.info("Accuracy(train) = {}".format(train_acc))
        logging.info("Accuracy(test) = {}".format(test_acc))

        # collect results for plotting for each epoch
        train_sizes.append(train_size)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot
    save_learning_curve(train_accs, test_accs, train_sizes)


if __name__ == '__main__':
    args = utils.bin_config(get_arguments)
    main()
