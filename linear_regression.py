import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os

def gradient_descent(x, y, theta, iterations, lr):
	m = x.shape[0]
	costs = []
	thetas = []
	for _ in range(0, iterations):
		prediction = np.dot(x, theta)
		error = prediction - y
		cost = (1/(2*m)) * np.dot(error.T, error)
		costs.append(cost)
		theta = theta - (lr * (1/m) * np.dot(x.T, error))
		thetas.append(theta)
	return thetas, costs


# Remove the NaNs from all the input features
# Normalize the inputs (mean 0 and std = 1): (x - x.mean()) / x.std()
# Else the cost decreasing function will not be smooth and be jaggedy

# Train set
x_train = stacked_train_features
y_train = train_labels.reshape(-1,1)

# Test Set
x_test = stacked_test_features
y_test = test_labels.reshape(-1,1)

# Add Bias unit to all the train, cv, and test set
bias_train = np.array( [1 for i in range(x_train.shape[0])] )
x_train = np.hstack(( bias_train.reshape(-1,1) , x_train ))

bias_test = np.array( [1 for i in range(x_test.shape[0])] )
x_test = np.hstack(( bias_test.reshape(-1,1) , x_test ))


# labels_dict= {0: 'unrelated', 1: 'discuss', 2: 'agree', 3: 'disagree'}
theta_initialized =  np.array( [random.uniform(-1,1) for i in range(x_train.shape[1])] ).reshape(-1,1)

num_iterations = 600
# learning_rate = 1.1
# learning_rate = 0.01
learning_rate = 0.5
thetas, costs = gradient_descent(x_train, y_train, theta_initialized, num_iterations, learning_rate)

# Showing the cost decreasing as # iterations increases
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Linear Regression (Learning Rate:' + str(learning_rate) +  ')')
plt.plot([t[0][0] for t in costs])
plt.show()

# Regression predictions
predictions = np.dot(x_test, thetas[-1])  # Shape is (m x 1)
predictions = predictions.reshape(predictions.shape[0])  # Removes the (1 column)
 
# Round the values to the nearest int (ex: 2.5 becomes label 2)
# Also remove all -ve signs and any label > 3 becomes label 3
predicted_LR_test_labels = np.abs( np.rint(predictions) )
predicted_LR_test_labels = np.array([p if p<=3 else 3 for p in predicted_LR_test_labels])


print ('Test Accuracy LR: ', sum(predicted_LR_test_labels == test_labels)/test_labels.shape[0])
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_LR_test_labels)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_LR_test_labels) )
print ('Test F1 Score: ', metrics.f1_score(test_labels, predicted_LR_test_labels, average='weighted'))
precision_recall( plot_confusion_matrix(test_labels, predicted_LR_test_labels)   )



