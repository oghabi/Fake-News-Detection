import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import os

# x is shape m x n (num_samples x features)
# y is m x 1
# theta is n x 1

# These are the prediction probabilities of the classes (1 or 0)
def hypothesis(theta, x):
	assert theta.shape[1] == 1  # theta is n x 1
	assert theta.shape[0] == x.shape[1]
	z = np.dot(x, theta)  # z = x * theta
	return 1. / (1 + np.exp(-z))

def cost(theta, x, y):
	assert x.shape[1] == theta.shape[0]  # x has a column for each feature, theta has a row for each feature.
	assert x.shape[0] == y.shape[0]  # One row per sample.
	assert y.shape[1] == 1
	assert theta.shape[1] == 1
	h = hypothesis(theta, x)
	one_case = np.matmul(-y.T, np.log(h))   # cost/penalty for y=1
	zero_case = np.matmul(-(1 - y).T, np.log(1 - h))  # cost/penalty for y=0
	return (one_case + zero_case) / x.shape[0]  # Normalize by number of samples

def gradient_descent(theta, x, y, learning_rate, regularization = 0):
	regularization = theta * regularization
	error = hypothesis(theta, x) - y
	n = (learning_rate / x.shape[0]) * (np.matmul(x.T, error) + regularization)
	return theta - n


def minimize(theta, x, y, iterations, learning_rate, regularization = 0):
	costs = []
	for _ in range(0, iterations):
		theta = gradient_descent(theta, x, y, learning_rate, regularization)
		costs.append(cost(theta, x, y)[0][0])
	return theta, costs


# Remove the NaNs from all the input features
# NORMALIZE the inputs (mean 0 and std = 1): (x - x.mean()) / x.std()
# Else the cost decreasing function will not be smooth and be jaggedy

# stacked_train_features = np.hstack(( train_KL.reshape(-1,1)  )).reshape(-1,1)
# stacked_cv_features = np.hstack(( cv_KL.reshape(-1,1) )).reshape(-1,1)
# stacked_test_features =  np.hstack((  test_KL.reshape(-1,1) )).reshape(-1,1)


stacked_train_features = np.hstack(( train_cosine_sim.reshape(-1,1) , train_tfidf.reshape(-1,1) , train_KL.reshape(-1,1),  train_BM25.reshape(-1,1) ))
stacked_cv_features = np.hstack((cv_cosine_sim.reshape(-1,1), cv_tfidf.reshape(-1,1), cv_KL.reshape(-1,1), cv_BM25.reshape(-1,1) ))
stacked_test_features =  np.hstack(( test_cosine_sim.reshape(-1,1), test_tfidf.reshape(-1,1), test_KL.reshape(-1,1),  test_BM25.reshape(-1,1) ))


# Scale each feature (0 mean unit std)
# scaled_features.mean(axis=0)  and  scaled_features.std(axis=0)
scaler = StandardScaler().fit(stacked_train_features)
stacked_train_features = scaler.transform(stacked_train_features)
stacked_cv_features = scaler.transform(stacked_cv_features)
stacked_test_features = scaler.transform(stacked_test_features)

# Train set
x_train = stacked_train_features
y_train = train_labels.reshape(-1,1)

# CV set
x_cv = stacked_cv_features
y_cv = cv_labels.reshape(-1,1)

# Test Set
x_test = stacked_test_features
y_test = test_labels.reshape(-1,1)

# Add Bias unit to all the train, cv, and test set
bias_train = np.array( [1 for i in range(x_train.shape[0])] )
x_train = np.hstack(( bias_train.reshape(-1,1) , x_train ))

bias_cv = np.array( [1 for i in range(x_cv.shape[0])] )
x_cv = np.hstack(( bias_cv.reshape(-1,1) , x_cv ))

bias_test = np.array( [1 for i in range(x_test.shape[0])] )
x_test = np.hstack(( bias_test.reshape(-1,1) , x_test ))


# Randomly initialize theta (1 for each feature) - Remember to include one for the bias unit as well
theta_initialized =  np.array( [random.uniform(-1,1) for i in range(x_train.shape[1])] ).reshape(-1,1)


# labels_dict= {0: 'unrelated', 1: 'discuss', 2: 'agree', 3: 'disagree'}

# Regularization Hyper-parameter tuning using CV set
for reg_param in np.linspace(0.01, 0.5, 50):
	thetas = []
	training_costs = []
	num_iterations = 600
	best_thetas = None  # Best list of thetas for the 4 logistic regression models
	best_cv_FNC_Score = 0
	best_reg_param = 0
	for i in range(0, len(labels_dict)):
		# Set the labels for this class as 1, and for the other classes as 0
		ovr_labels = np.array([1 if label==i else 0 for label in train_labels]).reshape(-1,1)
		# Save the thetas/weights for each trained logistic regression model in an array
		theta, costs = minimize(theta_initialized, x_train, ovr_labels, num_iterations, 1.1, reg_param)
		thetas.append(theta)
		training_costs.append(costs)

	# After creating 4 logistic regresssion models
	predicted_LR_cv = np.hstack(( hypothesis(thetas[0], x_cv), hypothesis(thetas[1], x_cv), hypothesis(thetas[2], x_cv), hypothesis(thetas[3], x_cv) ))
	predicted_LR_cv_labels = np.argmax(predicted_LR_cv, axis=1)
	cv_eval_score, cv_max_score = fnc_scorer(cv_labels, predicted_LR_cv_labels)
	cv_FNC_Score = cv_eval_score/cv_max_score

	if cv_FNC_Score > best_cv_FNC_Score:
		best_cv_FNC_Score = cv_FNC_Score
		best_thetas = thetas
		best_reg_param = reg_param


print ('Best regularization parameter: ', best_reg_param)
print ('Best FNC Score: ', best_cv_FNC_Score)


thetas = []
training_costs = []
num_iterations = 600
reg_param = 0.5
learning_rate = 0.5
# learning_rate = 0.01
# OvR (One vs Rest) for multi-class classification
# 4 Separate Logistic Regression Models
for i in range(0, len(labels_dict)):
	print ('OVR: Class: ', i, labels_dict[i], ' Vs. The Rest (class 0)' )
	# Set the labels for this class as 1, and for the other classes as 0
	ovr_labels = np.array([1 if label==i else 0 for label in train_labels]).reshape(-1,1)
	# Save the thetas/weights for each trained logistic regression model in an array
	theta, costs = minimize(theta_initialized, x_train, ovr_labels, num_iterations, learning_rate, reg_param)
	thetas.append(theta)
	training_costs.append(costs)



# Use the 4 logistic regression models to make the predictions (We saved the weights in thetas[])
predicted_LR_test = np.hstack(( hypothesis(thetas[0], x_test), hypothesis(thetas[1], x_test), hypothesis(thetas[2], x_test), hypothesis(thetas[3], x_test) ))

# This is One-vs-Rest method: For each sample, we have 4 probabilities (1 for each class), pick the maximum one for that sample
predicted_LR_test_labels = np.argmax(predicted_LR_test, axis=1)

print ('Test Accuracy LR: ', sum(predicted_LR_test_labels == test_labels)/test_labels.shape[0])
test_eval_score, test_max_score = fnc_scorer(test_labels, predicted_LR_test_labels)
print ('Test FNC Scorer: ', test_eval_score/test_max_score)
print ('Test confusion matrix: ', plot_confusion_matrix(test_labels, predicted_LR_test_labels) )
print ('Test F1 Score: ', metrics.f1_score(test_labels, predicted_LR_test_labels, average='weighted'))
precision_recall( plot_confusion_matrix(test_labels, predicted_LR_test_labels)   )


# Plot the cost as a function of the number of iterations
for j in range(0, len(labels_dict)):
	plt.xlabel('Iteration')
	plt.ylabel('Cost')
	plt.title('Logistic Regression (Learning Rate:' + str(learning_rate) +  ')')
	plt.plot(range(len(training_costs[j])), training_costs[j])
	plt.show()


























