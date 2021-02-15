import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"


ALGORITHM = "custom_net"


# ALGORITHM = "tf_net"

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


class NeuralNetwork_2Layer:
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        # Implemented sigmoid derivative
        f = self.__sigmoid(x)
        return f * (1 - f)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=5, minibatches=True, mbs=100):

        print('xShape: ', xVals.shape)
        print('yShape: ', yVals.shape)

        # for epoch
        for i in range(epochs):
            print('\nRunning epochs: ', i)
            number_of_batches = len(xVals) / mbs
            print('Number of batches being ran: ', number_of_batches)

            xTrainBatches = self.__batchGenerator(xVals, mbs)
            yTrainBatches = self.__batchGenerator(yVals, mbs)
            for j in range(int(len(xVals) / mbs)):

                # get current batches
                currentxTrainBatch = next(xTrainBatches)
                currentyTrainBatch = next(yTrainBatches)

                # forward
                (l1output, output) = self.__forward(currentxTrainBatch)

                # calculating error
                error = output - currentyTrainBatch
                l2d = error * self.__sigmoidDerivative(output)
                l1e = np.dot(l2d, self.W2.T)
                l1d = l1e * \
                    self.__sigmoidDerivative(
                        np.dot(currentxTrainBatch, self.W1))
                l1a = np.dot(currentxTrainBatch.T, l1d) * self.lr
                l2a = np.dot(l1output.T, l2d) * self.lr
                # print('layer 1 adj: ', l1a)
                # print('layer 2 adj: ', l2a)
                self.W1 = self.W1 - l1a
                self.W2 = self.W2 - l2a
                printProgressBar(j + 1, number_of_batches,
                                 prefix='Progress:', suffix='Complete', length=25)
        return None

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    def __back(self, input):
        pass

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    xTrain = xTrain / 255
    xTest = xTest / 255
    # reshape xtrain and xtest
    xTrainP = xTrain.reshape((60000, 784))
    xTestP = xTest.reshape((10000, 784))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        n = NeuralNetwork_2Layer(len(xTrain[0]), len(yTrain[0]), 512)
        n.train(xTrain, yTrain, epochs=3)
        return n
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")

        model = keras.models.Sequential([keras.layers.Flatten(), keras.layers.Dense(
            768, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])
        opt = keras.optimizers.Adam(learning_rate=0.0009)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=15)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        pred = model.predict(data)
        return pred
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):  # TODO: Add F1 score confusion matrix here.
    if ALGORITHM == "guesser":
        xTest, yTest = data
        acc = 0
        print(preds.shape)
        print(yTest.shape)
        for i in range(preds.shape[0]):
            if np.array_equal(preds[i], yTest[i]):
                acc = acc + 1
        accuracy = acc / preds.shape[0]
        print("Classifier algorithm: %s" % ALGORITHM)
        print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()
    elif ALGORITHM == "tf_net" or ALGORITHM == "custom_net":
        xTest, yTest = data
        preds = np.argmax(preds, axis=1)
        yTest = np.argmax(yTest, axis=1)
        f1 = np.zeros((10, 10))
        for (val1, val2) in zip(preds, yTest):
            f1[val1][val2] = f1[val1][val2] + 1

        print('F1 matrix:')
        np.set_printoptions(suppress=True)
        header = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        format_row = "{:>12}" * (len(header) + 1)
        print(format_row.format("", *header))
        for h, v in zip(header, f1):
            print(format_row.format(h, *v))

        total_pos = sum(preds == yTest)
        accuracy = total_pos / len(preds)
        print("Classifier algorithm: %s" % ALGORITHM)
        print("Classifier accuracy: %f%%" % (accuracy * 100))
        print()


# =========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == '__main__':
    main()
