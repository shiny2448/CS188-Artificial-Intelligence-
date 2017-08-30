import numpy as np
import math
import util
import matplotlib.pyplot as plt
from util import raiseNotDefined


def display_digit_features(weights, bias):
    """Visualizes a set of weight vectors for each digit.
        Do not modify this code."""
    feature_matrices = []
    #weights is 1000*784 matrix, and is zero matrix
    # counter=0
    # for i in range(10):
    #     print 'weiths.shape[0]:',weights.shape[0]
    #     print 'weithts.shape[1]:',weights.shape[1]
    #     for j in range(weights.shape[0]):
    #         if weights[j,i]!=0:
    #             counter+=1
    # print 'counter in display_digit_features:',counter
    for i in range(10):
        # for j in range(weights[:,i].shape[0]):
        #     if weights[j,i]!=0:
        #         counter+=1
        # print 'counter in display_digit_features:',counter
        # counter=0
      #print 'weights[:,i]',weights[:,i]
        feature_matrices.append(convert_weight_vector_to_matrix(weights[:, i], 28, 28, bias))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(feature_matrices[i], cmap='gray')
        
    plt.show()

def apply_bias(samples):
    """
    samples: The samples under test, should be a numpy array of shape (numSamples, numFeatures).
    Mutates samples to add a bias term to each feature vector. This example appends a 1 to
    the front, but the bias term could be added anywhere.

    Do not modify this code."""
    return np.hstack([samples, np.ones((len(samples), 1))])

def simple_image_featurization(image):
    """Converts an image to a numpy vector of shape (1, w * h), where w is the
        width of the image, and h is the height."""

    """YOUR CODE HERE"""
    #raiseNotDefined()
    #print 'image: ', image
    #print 'image.getPixel: ', image.getPixel(0,0)
    vector = np.zeros((28*28))
    i=0
    for row in range(28):
        for col in range(28):
            vector[i] = image.getPixel(col,row)
            i+=1
    return vector

def zero_one_loss_ss(classifier, sample, label):
    """
    classifier: The classifier under test.
    sample: The sample under test, should be a numpy array of shape (1, numFeatures).
    label: The correct label of the sample under test.

    Returns 0.0 if the classifier classifies the sample correctly, or 1.0 otherwise."""

    """YOUR CODE HERE"""
    #raiseNotDefined()
    #print "classifier:", classifier
    #print "samples:", sample
    #print "label:", label
    #scores_classifier = np.dot()
    if classifier.classify(sample)==label:
        return 0.0
    else:
        return 1.0



def zero_one_loss(classifier, samples, labels):
    """
    classifier: The classifier under test.
    sample: The samples under test, should be a numpy array of shape (numSamples, numFeatures).
    label: The correct labels of the samples under test.

    Returns the fraction of samples that are wrong. For example, if the classifier gets
    65 out of 100 samples right, this function should return 0.35."""

    """YOUR CODE HERE"""
    #raiseNotDefined()
    correct_count = 0.0
    for i in range(samples.shape[0]):
        if zero_one_loss_ss(classifier, samples[i], labels[i]):
            correct_count += 1.0
    return 1.0-correct_count/samples.shape[0]



def convert_weight_vector_to_matrix(weight_vector, w, h, bias):
    """weight_vector: The weight vector to transformed into a matrix.
    w: the width of the matrix
    h: the height of the matrix
    bias: whether or not there is a bias feature

    Returns a w x h array where the first w entries of the weight vector for this label correspond to the
    first row, the next w the next row, and so forth. Assume that w * h is equal to the size of the
    weight vector. Ignore the bias if there is one"""

    """YOUR CODE HERE"""
    #raiseNotDefined()
    #return weight_vector.resize((w,h))
    #return np.reshape(weight_vector, (w,h))
    #print 'weight_vector: ', weight_vector
    #weight_vector = weight_vector.transpose()
    matrix = np.zeros((w,h))
    weight_vector_index = 0
    #print 'bias:',bias
    #print 'weight_vector.shape[0]', weight_vector.shape[0] #1000
    #if bias:
    #    weight_vector = weight_vector[1:weight_vector.shape[0]-1]
    for width in range(w):
        for height in range(h):
            if weight_vector_index < weight_vector.shape[0]:
                matrix[width,height] = weight_vector[weight_vector_index]
                weight_vector_index += 1
    #counter=0
    #for i in range(weight_vector.shape[0]):
    #    if weight_vector[i]!=0:
    #        counter+=1
    #for width in range(w):
    #    for height in range(h):
    #        if matrix[width,height]!=0:
    #            counter+=1
    #        weight_vector_index2+=1
    #print 'counter:',counter
    return matrix
