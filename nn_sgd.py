import numpy as np
import pandas as pd

class NeuralNetwork():
    def __init__(self, width):
        self.width = width
        self.weight_vector_1 = np.array([]) 
        self.bais_1 = np.array([]) 
        self.weight_vector_2 = np.array([]) 
        self.bias_2 = np.array([]) 
        self.weight_vector_3 = np.array([]) 
        self.bias_3 = np.array([]) 

    def weight(self, d, initialize_random=True):
        self.weight_vector_1 = np.random.normal(size=(d,self.width)) if initialize_random else np.zeros((d,self.width))
        self.bais_1 = np.random.normal(size=(self.width)) if initialize_random else np.zeros((self.width))
        self.weight_vector_2 = np.random.normal(size=(self.width,self.width)) if initialize_random else np.zeros((self.width,self.width)) 
        self.bias_2 = np.random.normal(size=(self.width)) if initialize_random else np.zeros((self.width)) 
        self.weight_vector_3 = np.random.normal(size=(self.width,1)) if initialize_random else np.zeros((self.width,1)) 
        self.bias_3 = np.random.normal(size=(1)) if initialize_random else np.zeros((1)) 

    
    def forward_pass(self, X):
        S1 = np.dot(X, self.weight_vector_1) + self.bais_1 
        Z1 = self.sigmoid(S1) 

        S2 = np.dot(Z1, self.weight_vector_2) + self.bias_2 
        Z2 = self.sigmoid(S2) 

        output = np.dot(Z2, self.weight_vector_3) + self.bias_3 
        calculated_values = (S1, Z1, S2, Z2)

        return output, calculated_values

    def backward_propogation(self, X, y, output, calculated_values):
        S1, Z1, S2, Z2 = calculated_values

        dy = (output - y).reshape((1,1)) 
        dweight_vector_3 = np.dot(Z2.reshape((1,-1)).T, dy) 
        dbias_3 = np.sum(dy, axis=0) 
        dZ2 = np.dot(dy, self.weight_vector_3.reshape((-1,1)).T) 

        dsigmoid_2 = self.sigmoid(S2) * (1 - self.sigmoid(S2)) * dZ2 
        dweight_vector_2 = np.dot(Z1.reshape((1,-1)).T, dsigmoid_2) 
        dbias_2 = np.sum(dsigmoid_2, axis=0) 
        dZ1 = np.dot(dsigmoid_2, self.weight_vector_2.T) 

        dsigmoid_1 = self.sigmoid(S1) * (1 - self.sigmoid(S1)) * dZ1
        dweight_vector_1 = np.dot(X.reshape((1,-1)).T, dsigmoid_1)
        dbias_1 = np.sum(dsigmoid_1, axis=0) 
        
        return dweight_vector_1, dbias_1, dweight_vector_2, dbias_2, dweight_vector_3, dbias_3


    def train(self, X, y, T, threshold, default_learning_rate, initialize_random=True, initial_learning_rate=None):
        num_samples, num_features = X.shape
        self.weight(num_features, initialize_random)

        indices = np.arange(num_samples)
        current_error = 0
        for t in range(T):
            np.random.shuffle(indices) 
            learning_rate_t = default_learning_rate if initial_learning_rate is None else initial_learning_rate[t]
            for i in indices:
                x = X[i,:].reshape((1,-1)) 
                output, calculated_values = self.forward_pass(x)

                dweight_vector_1, dbias_1, dweight_vector_2, dbias_2, dweight_vector_3, dbias_3 = self.backward_propogation(x, y[i], output, calculated_values)

                self.weight_vector_1 -= learning_rate_t * dweight_vector_1
                self.bais_1 -= learning_rate_t * dbias_1 
                self.weight_vector_2 -= learning_rate_t * dweight_vector_2
                self.bias_2 -= learning_rate_t * dbias_2 
                self.weight_vector_3 -= learning_rate_t * dweight_vector_3 
                self.bias_3 -= learning_rate_t * dbias_3 

            forward_pass_output, _ = self.forward_pass(X)

            if (abs(current_error - self.mean_squared_error(forward_pass_output, y))) < threshold:
                break


    def fit(self, X):
        forawrd_pass_output, _ = self.forward_pass(X)
        return np.sign(forawrd_pass_output.flatten())
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def mean_squared_error(self, output, y):
        return np.mean(0.5 * ((output - y)**2))

def prediction_helper(y):
    ycp = y.copy()
    ycp[ycp==0] = -1
    return ycp

if __name__ == "__main__":
    df_train = pd.read_csv('/Users/u1503285/CS-6350-ML/NeuralNetworks/bank-note/train.csv', header=None)
    df_test = pd.read_csv('/Users/u1503285/CS-6350-ML/NeuralNetworks/bank-note/test.csv', header=None)
    df_train_X = df_train.iloc[:, 0:4].values
    df_train_y = prediction_helper(df_train.iloc[:, 4].values)
    df_test_X = df_test.iloc[:, 0:4].values
    df_test_y = prediction_helper(df_test.iloc[:, 4].values)

    t = np.arange(100)
    learning_rate = 0.1 / (1 + (0.1/0.01)*t)
    width_list = np.array([5, 10, 25, 50, 100])

    print("Weights initialized with random in Gaussian Distribution\n")
    for width in width_list:
        net = NeuralNetwork(width)
        net.train(df_train_X, df_train_y, T=100, threshold=1e-9, default_learning_rate=0.1, initialize_random=False, initial_learning_rate=learning_rate)

        train_pred = net.fit(df_train_X)
        train_err = np.sum(train_pred!=df_train_y) / len(df_train_y)
        test_pred = net.fit(df_test_X)
        test_err = np.sum(test_pred!=df_test_y) / len(df_test_y)
        print("Width: ",width)
        print("Train error: ",train_err)
        print("Test error: ",test_err)

    t = np.arange(100)+1
    learning_rate = 0.1 / (1 + (0.1/0.1)*t)
    width_list = np.array([5, 10, 25, 50, 100])

    print("Weights initialized with zero\n")
    for width in width_list:
        net = NeuralNetwork(width)
        e = net.train(X=df_train_X, y=df_train_y, T=100, threshold=1e-9, default_learning_rate=0.1, initialize_random=False, initial_learning_rate=learning_rate)

        train_pred = net.fit(df_train_X)
        train_err = np.sum(train_pred!=df_train_y) / len(df_train_y)
        test_pred = net.fit(df_test_X)
        test_err = np.sum(test_pred!=df_test_y) / len(df_test_y)
        print("Width: ",width)
        print("Train error: ",train_err)
        print("Test error: ",test_err)