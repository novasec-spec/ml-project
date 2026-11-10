
import numpy as np

X = np.array([
   [0, 0],
   [0, 1],
   [1, 0],
   [1, 1]
], dtype=float)

y = np.array([ 
 [0],
 [1],
 [1],
 [0]
], dtype=float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(x):
   return x * (1 -x)
 


np.random.seed(42)


W1 = np.random.randn(2, 4)

W2 = np.random.randn(4, 1)


b1 = np.zeros((1, 4))
b2 = np.zeros((1, 1))
  
learning_rate = 0.5



for epoch in range(10000):



    hidden_input = X @ W1 + b1
hidden_output = sigmoid(hidden_input)

output_input = hidden_output @ W2
+ b2

prediction = sigmoid(output_input)


error = y - prediction 

if epoch % 1000 == 0:
    loss = np.mean(error ** 2)


output_gradient = error * sigmoid_derivative(prediction)

hidden_error = output_gradient @ W2.T

hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)

W2 += hidden_output.T @ output_gradient * learning_rate
b2 += np.sum(output_gradient,
axis=0, keepdims=True) * learning_rate

W1 += X.T @ hidden_gradient * learning_rate 
b1 += np.sum(hidden_gradient,
axis=0, keepdims=True) * learning_rate

print("\nAI predictions:\n")

for inputs, answer in zip(X, y):
    hidden = sigmoid(inputs @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)

    print(
         f"{inputs.astype(int)} -> "
         f"{output[0][0]:.4f} "
         f"(expected {int(answer[0])})"
)
