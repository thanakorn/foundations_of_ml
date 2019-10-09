#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#%% Generate Samples
num_data_per_class = 200

m1 = [[0,5]]
m2 = [[5,0]]
C  = [[2,1], [1,2]]

A = np.linalg.cholesky(C)

X1 = np.random.randn(num_data_per_class, 2)
Y1 = (X1 @ A) + m1

X2 = np.random.randn(num_data_per_class, 2)
Y2 = (X2 @ A) + m2

plt.figure(figsize=(3,3))
plt.scatter(Y1[:,0], Y1[:,1], c = 'r', label = 'Y1')
plt.scatter(Y2[:,0], Y2[:,1], c = 'b', label = 'Y2')
plt.legend()
plt.savefig('scatter.png')
#%% Generate training set and testing set
samples = np.concatenate((Y1, Y2))
label_positive = np.ones(num_data_per_class)
label_negative = np.ones(num_data_per_class)
labels = np.concatenate((label_positive, label_negative))

sample_indices = np.random.permutation(num_data_per_class * 2)
train_samples = samples[sample_indices, :]
train_labels = labels[sample_indices]

# Check accuracy
def percent_correct(inputs, targets, weights):
    n = len(targets)
    num_correct = 0
    for i in range(n):
        input = inputs[i,:]
        if(targets[i] * np.dot(input, weights) > 0):
            num_correct += 1
    return (num_correct / n) * 100

# Learning
#weights = np.random.randn(2)# Initiate weights by random
weights = np.random.randn(3)
print('Initial weights :', weights)
print('Initial weights accuracy : ', percent_correct(data_train, label_train, weights))

num_train = 200
iteration = 1000
learning_rate = 0.01

accuracy_train = np.zeros(iteration)
accuracy_test = np.zeros(iteration)

for i in range(iteration):
    random_index = np.floor(np.random.rand() * num_train).astype(int)
    sample = data_train[random_index,:]
    
    # If misclassified, adjust the weight based on how much the inaccuracy is
    if(label_train[random_index] * np.dot(sample, weights) < 0):
        weights += learning_rate * label_train[random_index] * sample
        
    accuracy_train[i] = percent_correct(data_train, label_train, weights)
    accuracy_test[i] = percent_correct(data_test, label_test, weights)
    
plt.plot(range(iteration), accuracy_train, c = 'r', label = 'accuracy_train')
plt.plot(range(iteration), accuracy_test, c = 'g', label = 'accuracy_test')
plt.legend()

print('Final weights : ', weights)
print('Final weights train accuracy : ', accuracy_train[-1])
print('Final weights test accuracy : ', accuracy_test[-1])

# Perceptron using scikit-learn
classifier = Perceptron()
classifier.fit(data_train, label_train)
train_predict = classifier.predict(data_train)
test_predict = classifier.predict(data_test)
print('Scikit-learn train accuracy : ', accuracy_score(train_predict, label_train) * 100)
print('Scikit-learn test accuracy : ', accuracy_score(test_predict, label_test) * 100)
