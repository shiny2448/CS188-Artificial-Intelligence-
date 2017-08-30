import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        #raiseNotDefined()
        self.weights={}
        for category in self.categories:
          self.weights[category] = [0 for _ in range(numFeatures)]
    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        #raiseNotDefined()
        scores = {}
        for category in self.categories:
          scores[category] = np.dot(self.weights[category],sample)
        return max(scores, key=scores.get)
        # max_score = np.dot(self.weights[0],sample)
        # max_category = self.categories[0]
        # for i in range(len(self.categories)):
        #   curr_score = np.dot(self.weights[i],sample)
        #   if curr_score > max_score:
        #     max_score = curr_score
        #     max_category = self.categories[i]
        # return max_category

    def train(self, samples, labels):
        """samples: np.array of shape (numSamples, numFeatures)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        #raiseNotDefined()
        #print 'train function: labels:',labels
        for i in range(samples.shape[0]):
          y_prime = self.classify(samples[i]) #prediction
          y_star = labels[i] #true label
          if y_prime!=y_star:
            self.weights[y_prime] -= samples[i]
            self.weights[y_star] += samples[i]
        # for i in range(samples.shape[0]):
        #   y_prime = self.classify(samples[i])
        #   y_star = labels[i]
        #   if y_prime != y_star:
        #     y_prime_index = self.weights_index_dict[y_prime]
        #     y_star_index = self.weights_index_dict[y_star]
        #     #print 'samples[i]:'
        #     self.weights[y_prime_index] -= samples[i]
        #     self.weights[y_star_index] += samples[i]


