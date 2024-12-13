import numpy as np
from nn_sgd import NeuralNetwork

forward_pass_result_by_hand = np.array([[-2.437]])
forward_pass_neurons_by_hand = (np.array([[-6, 6]]), np.array([[0.00247, 0.9975]]), np.array([[-4, 4]]), np.array([[0.01803, 0.9820]]))

backward_propogation_byhand= np.array([[0.00101, 0.00153], [0.00101, 0.00153]]), np.array([0.00101, 0.00153]), np.array([[-0.0003, 0.00022], [-0.121, 0.0916]]), np.array([-0.121, 0.0916]), np.array([[-0.0618], [-3.3746]]), np.array([-3.4368])

neural_network_3 = NeuralNetwork(2)
X_input = np.array([[1,1]]) 
y_output = np.array([1])
neural_network_3.weight_vector_1 = np.array([[-2,2],[-3,3]])
neural_network_3.bais_1 = np.array([-1, 1])
neural_network_3.weight_vector_2 = np.array([[-2,2],[-3,3]]) 
neural_network_3.bias_2 = np.array([-1, 1]) 
neural_network_3.weight_vector_3 = np.array([[2],[-1.5]]) 
neural_network_3.bias_3 = np.array([-1]) 

forward_pass_result, forward_pass_values = neural_network_3.forward_pass(X_input)
backward_pass_values = neural_network_3.backward_propogation(X_input, y_output, forward_pass_result, forward_pass_values)

print('Forward pass by hand:')
print('Score',forward_pass_result_by_hand)
print('neuron layer 1:',forward_pass_neurons_by_hand[1])
print('neuron layer 2:',forward_pass_neurons_by_hand[3])
print('\nForward pass results:')
print('Score',forward_pass_result)
print('neuron layer 1:',forward_pass_values[1])
print('neuron layer 2:',forward_pass_values[3])

print('\nBackward pass by hand:')
print('d weight vector 1:',backward_propogation_byhand[0])
print('d bias 1:',backward_propogation_byhand[1])
print('d weight vector 2:',backward_propogation_byhand[2])
print('d bias 2:',backward_propogation_byhand[3])
print('d weight vector 3:',backward_propogation_byhand[4])
print('d bias 3:',backward_propogation_byhand[5])

print('\nBackward pass results:')
print('d weight vector 1:',backward_pass_values[0])
print('d bias 1:',backward_pass_values[1])
print('d weight vector 2:',backward_pass_values[2])
print('d bias 2:',backward_pass_values[3])
print('d weight vector 3:',backward_pass_values[4])
print('d bias 3:',backward_pass_values[5])
