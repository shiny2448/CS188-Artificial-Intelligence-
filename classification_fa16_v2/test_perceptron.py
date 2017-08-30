import numpy as np
import perceptron
import samples
import data_classification_utils as dcu
from util import raiseNotDefined

"""feel free to play with these values and see what happens"""
bias = False
num_times_to_train = 5 #10
num_train_examples = 1000 #5000

def get_perceptron_training_data():
	training_data = samples.loadDataFile("digitdata/trainingimages", num_train_examples, 28, 28)
	training_labels = map(str, samples.loadLabelsFile("digitdata/traininglabels", num_train_examples))

	featurized_training_data = np.array(map(dcu.simple_image_featurization, training_data))
	return training_data, featurized_training_data, training_labels

def get_perceptron_test_data():
	test_data = samples.loadDataFile("digitdata/testimages", 1000, 28,28)
	test_labels = map(str, samples.loadLabelsFile("digitdata/testlabels", 1000))

	featurized_test_data = np.array(map(dcu.simple_image_featurization, test_data))
	return test_data, featurized_test_data, test_labels

"""
if you want a bias, then apply that bias to your data, then create a perceptron to identify digits

Next, train that perceptron on the entire set of training data num_times_to_train times on num_train_examples.

Finally, use the zero_one_loss defined in data_classification_utils to find the 
final accuracy on both the training set and the test set, assigning them to the 
variables training_accuracy and test_accuracy respectively"""



raw_training_data, featurized_training_data, training_labels = get_perceptron_training_data()
raw_test_data, featurized_test_data, test_labels = get_perceptron_test_data()

"""YOUR CODE HERE"""
#print 'debug',featurized_training_data.shape
#print 'labes:',len(training_labels)
#print 'featurized_training_data.shape[1]:',featurized_training_data.shape[1]
new_perceptron = perceptron.Perceptron(['0','1','2','3','4','5','6','7','8','9'], featurized_training_data.shape[1])

#if bias:
#	featurized_training_data = data_classification_utils.apply_bias(featurized_training_data)
#print 'type of raw_training_data:', type(raw_training_data) #list
#print 'type of featurized_training_data:', type(featurized_training_data) #list

# counter=0
# for i in range(featurized_training_data.shape[0]):
# 	for j in range(featurized_training_data.shape[1]):
# 		#print 'featurized_training_data[i,j]:',featurized_training_data[i,j]
# 		#print 'raw_training_data[i][j]:',raw_training_data[i][j]
# 		if featurized_training_data[i,j]!=raw_training_data[i][j]: 
# 			counter+=1
# print 'counter:',counter
if bias:
	featurized_training_data = data_classification_utils.apply_bias(featurized_training_data)
for train_index in range(num_times_to_train):
	featurized_training_data = featurized_training_data[0:min(featurized_training_data.shape[0],num_train_examples),:]
	new_perceptron.train(featurized_training_data, training_labels)
#print 'new_perceptron.weights:',new_perceptron.weights
# counter=0
# for i in range(new_perceptron.weights.shape[0]):
#  	for j in range(new_perceptron.weights.shape[1]):
#  		if new_perceptron.weights[i,j]!=0.0:
#  			counter+=1
# print 'counter:',counter
# counter=0
training_accuracy = dcu.zero_one_loss(new_perceptron,featurized_training_data, training_labels)*100
test_accuracy = dcu.zero_one_loss(new_perceptron, featurized_test_data, test_labels)*100
# print 'after computing accuracy:'
# for i in range(new_perceptron.weights.shape[0]):
#  	for j in range(new_perceptron.weights.shape[1]):
#  		if new_perceptron.weights[i,j]!=0.0:
#  			counter+=1
# print 'counter:',counter
#training_accuracy = None
#test_accuracy = None
print('Final training accuracy: ' + str(training_accuracy) + '% correct')

print("Test accuracy: " + str(test_accuracy) + '% correct')

dcu.display_digit_features(dcu.convert_perceptron_weights_to_2D_array_with_ten_columns(new_perceptron), bias)