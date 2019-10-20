#%% Import libraries
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
plt.rcParams.update(plt.rcParamsDefault)

#%% Calculate accuracy
def percent_correct(inputs, targets, w):
    N = len(targets)
    num_correct = 0
    for i in range(N):
        input = inputs[i,:]
        if(targets[i] * np.dot(input, w) > 0):
            num_correct += 1
    return 100 * num_correct / N

#%% Learning method
def learn_perceptron(samples, sample_labels, test, test_classes, learning_rate = 0.01, iteration = 1000):
    num_data = samples.shape[0]
    num_attibute = samples.shape[1]
    weights = np.random.randn(num_attibute)
    accuracy = np.zeros(iteration)
    accuracy_test = np.zeros(iteration)
    for i in range(iteration):
        random_index = np.floor(np.random.rand() * num_data).astype(int)
        sample = samples[random_index,:]
    
        # If misclassified, adjust the weight based on how much the inaccuracy is
        if(sample_labels[random_index] * np.dot(sample, weights) < 0):
            weights += learning_rate * sample_labels[random_index] * sample
        accuracy[i] = percent_correct(samples, sample_labels, weights)
        accuracy_test[i] = percent_correct(test, test_classes, weights)

    plt.plot(range(iteration), accuracy, c = 'r', label = 'Training set')
    plt.plot(range(iteration), accuracy_test, c = 'g', label = 'Test set')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    return weights
#%% Plot decision boundary
def plot_decision_boundary(weights):
    m = -weights[0] / weights[1]
    x = np.linspace(0,5)
    y = m * x - (weights[2]) / weights[1]
    plt.plot(x,y,c='k')

#%% Testing with Iris datasets
iris_data_raw = pd.read_csv('/home/tpanyapiang/git/MSc/foundations_of_ml/Lab2/iris.data',
                        names=['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'name'])
iris_labels = iris_data_raw['name'].unique()[:2]
iris_data = iris_data_raw[iris_data_raw['name'].isin(iris_labels)] # Select only 2 classes
iris_data['label'] = iris_data['name'].apply(lambda x: 1 if x == iris_labels[0] else -1) # Convert text to class
iris_data['bias'] = 1

#%% Visualize data
setosa = iris_data[iris_data['name'] == 'Iris-setosa']
versicolor =  iris_data[iris_data['name'] == 'Iris-versicolor']
plt.figure(figsize=(5,5))
plt.scatter(setosa['sepal_length_cm'].values, setosa['sepal_width_cm'].values, c = 'r', label = 'setosa')
plt.scatter(versicolor['sepal_length_cm'].values, versicolor['sepal_width_cm'].values, c = 'b', label = 'versicolor')
plt.xlabel('sepal_length_cm')
plt.ylabel('sepal_width_cm')
plt.legend()
plt.savefig('Lab2/sepal_compare.png',bbox='tight')
plt.show()
plt.figure(figsize=(5,5))
plt.scatter(setosa['petal_length_cm'].values, setosa['petal_width_cm'].values, c = 'r', label = 'setosa')
plt.scatter(versicolor['petal_length_cm'].values, versicolor['petal_width_cm'].values, c = 'b', label = 'versicolor')
plt.xlabel('petal_length_cm')
plt.ylabel('petal_width_cm')
plt.legend()
plt.savefig('Lab2/petal_compare.png',bbox='tight')
plt.show()

#%% Split data
iris_train, iris_test = train_test_split(iris_data, test_size = 0.7, shuffle = True)
print('Num data : ', iris_data.shape[0])
print('Num train : ', iris_train.shape[0])
print('Num test : ', iris_test.shape[0])

iris_train_labels = iris_train['label'].to_numpy()
iris_test_labels = iris_test['label'].to_numpy()

#%% Train using sepal
iris_sepal_train = iris_train[['sepal_length_cm','sepal_width_cm','bias']].to_numpy()
iris_sepal_test = iris_test[['sepal_length_cm','sepal_width_cm','bias']].to_numpy()
plt.figure()
iris_weights_sepal = learn_perceptron(iris_sepal_train, iris_train_labels, iris_sepal_test, iris_test_labels)
plt.title('Train using sepal length & width')
# plt.savefig('Lab2/iris_sepal_learning_.png')
plt.show()
print('Trained sepal weights : ', iris_weights_sepal)
sepal_test_acc = percent_correct(iris_sepal_test, iris_test_labels, iris_weights_sepal)
print('Accuracy on sepal test set : ', sepal_test_acc)
plt.figure(figsize=(4,4))
plt.scatter(setosa['sepal_length_cm'].values, setosa['sepal_width_cm'].values, c = 'r', label = 'setosa')
plt.scatter(versicolor['sepal_length_cm'].values, versicolor['sepal_width_cm'].values, c = 'b', label = 'versicolor')
plt.xlabel('sepal_length_cm')
plt.ylabel('sepal_width_cm')
plot_decision_boundary(iris_weights_sepal)
plt.legend()
plt.savefig('Lab2/sepal_decision.png',bbox='tight')

#%% Train using petal
iris_petal_train = iris_train[['petal_length_cm','petal_width_cm', 'bias']].to_numpy()
iris_petal_test = iris_test[['petal_length_cm','petal_width_cm', 'bias']].to_numpy()
plt.figure()
iris_weights_petal = learn_perceptron(iris_petal_train, iris_train_labels, iris_petal_test, iris_test_labels)
plt.title('Train using petal length & width')
#plt.savefig('Lab2/iris_petal_learning_.png')
plt.show()
plt.figure(figsize=(4,4))
plt.scatter(setosa['petal_length_cm'].values, setosa['petal_width_cm'].values, c = 'r', label = 'setosa')
plt.scatter(versicolor['petal_length_cm'].values, versicolor['petal_width_cm'].values, c = 'b', label = 'versicolor')
plt.xlabel('petal_length_cm')
plt.ylabel('petal_width_cm')
plot_decision_boundary(iris_weights_petal)
plt.legend()
plt.savefig('Lab2/petal_decision.png',bbox='tight')

#%% Train using all feature
iris_features_train = iris_train[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm', 'bias']].to_numpy()
iris_features_test = iris_test[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm', 'bias']].to_numpy()
plt.figure()
iris_weights = learn_perceptron(iris_features_train, iris_train_labels, iris_features_test, iris_test_labels)
plt.title('Train using petal length & width')
#plt.savefig('Lab2/iris_learning_.png')
plt.show()

#%%
