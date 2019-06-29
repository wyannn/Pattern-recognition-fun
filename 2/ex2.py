import numpy as np
import matplotlib.pyplot as plt
import xlrd as xd
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import tree
from sigmoid import sigmoid 
from evaluate import evaluate
from feature_normalize import feature_normalize
from initialize_parameters import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost  
from backward_propagation import backward_propagation
from update_parameters import update_parameters
from predict import predict
from randomized_data import randomized_data

#读取数据（训练集）
data = xd.open_workbook('data.xls')
table = data.sheets()[0]
sex_men = []
sex_women = []
sex = table.col_values(1)[1:]
height = table.col_values(3)[1:]
weight = table.col_values(4)[1:]
likemath = table.col_values(6)[1:]
likeart = table.col_values(7)[1:]
likesport = table.col_values(8)[1:]

data = np.vstack((sex, height, weight, likemath, likeart, likesport))

index = randomized_data(sex)
data_x_test = np.zeros((6,1))

for i in index:
    data_x_test = np.column_stack((data_x_test, data[:,i]))
data_x_test = np.delete(data_x_test, 0, axis = 1)
data_x = np.delete(data, index, axis = 1)

X = data_x[1:,:]
m = X.shape[1]
Y = data_x[0,:].reshape(1, m)

X_test = data_x_test[1:,:]
m2 = X_test.shape[1]
Y_test = data_x_test[0,:].reshape(1, m2)

shape_X = X.shape
shape_Y = Y.shape

X = feature_normalize(X)
X_test = feature_normalize(X_test)


def layer_sizes(X, Y):
    n_x = X.shape[0] 
    n_h = 5
    n_y = Y.shape[0] 
    return (n_x, n_h, n_y)


def nn_model(X, Y, n_h, num_iterations = 1500, print_cost=False):
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        
        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y, parameters)
 
        grads = backward_propagation(parameters, cache, X, Y)
 
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        plt.scatter(i+1, cost)
        plt.title('cost curve')
        plt.xlabel('iteration times')
        plt.ylabel('cost')
    plt.savefig('cost curve.jpg')
    return parameters



#神经网络
parameters = nn_model(X, Y, n_h = 5, num_iterations = 1500, print_cost=True)
predictions = predict(parameters, X_test)
SE1, SP1, ACC1 = evaluate(predictions, Y_test)
AUC1 = roc_auc_score(Y_test.T, predictions.T)
print('\n' + '-' * 20)
print('BP神经网络分类结果评估:')
print('SE: %s' % SE1)
print('SP: %s' % SP1)
print('ACC: %s' % ACC1)
print('AUC: %s' % AUC1)
print('-'*20 + '\n') 

#支持向量机
clf = svm.SVC(gamma = 10)    
clf.fit(X.T, Y.reshape(Y.shape[1],))
predictions = clf.predict(X_test.T)
SE2, SP2, ACC2 = evaluate(predictions, Y_test.reshape(Y_test.shape[1],))
AUC2 = roc_auc_score(Y_test.T, predictions.T)
print('\n' + '-' * 20)
print('SVM分类结果评估:')
print('SE: %s' % SE2)
print('SP: %s' % SP2)
print('ACC: %s' % ACC2)
print('AUC: %s' % AUC2)
print('-'*20 + '\n') 

#决策树
clf = tree.DecisionTreeClassifier(criterion='entropy') 
clf.fit(X.T, Y.T)
predictions = clf.predict(X_test.T)
SE3, SP3, ACC3 = evaluate(predictions, Y_test.reshape(Y_test.shape[1],))
AUC3 = roc_auc_score(Y_test.T, predictions.T)
print('\n' + '-' * 20)
print('决策树分类结果评估:')
print('SE: %s' % SE3)
print('SP: %s' % SP3)
print('ACC: %s' % ACC3)
print('AUC: %s' % AUC3)
print('-'*20 + '\n') 
