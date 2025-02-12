# python_neural_network_activation_function

This repository contains implementations of various neural network activation functions in Python using NumPy. The activation functions included are:
- Step function
- Sigmoid function
- Hyperbolic Tangent (tanh) function
- ReLU (Rectified Linear Unit) function
- Softmax function
- Linear function

## Installation and Usage

To use these activation functions, ensure you have Python and NumPy installed. You can install NumPy using pip:

```bash
pip install numpy
```

You can then import and use the functions in your Python scripts or Jupyter notebooks.

## Examples

### Step Function

```python
import numpy as np

def step_function(x):
    return 1 * (x > 0)

print(step_function(-10))  # Output: 0
print(step_function(10))   # Output: 1
```

### Sigmoid Function

```python
import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid_function(-10))  # Output: ~0.000045
print(sigmoid_function(10))   # Output: ~0.999955
print(sigmoid_function(1))    # Output: ~0.731059
```

### Hyperbolic Tangent Function

```python
import numpy as np

def hyperbolic_tangent_function(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

print(hyperbolic_tangent_function(-10))  # Output: ~-1
print(hyperbolic_tangent_function(10))   # Output: ~1
print(hyperbolic_tangent_function(1))    # Output: ~0.761594
```

### ReLU Function

```python
import numpy as np

def relu_function(x):
    return np.maximum(0, x)

print(relu_function(-10))  # Output: 0
print(relu_function(10))   # Output: 10
print(relu_function(1))    # Output: 1
```

### Softmax Function

```python
import numpy as np

def softmax_function(x):
    return np.exp(x) / np.sum(np.exp(x))

print(softmax_function([1, 2, 3]))  # Output: array([0.09003057, 0.24472847, 0.66524096])
```

### Linear Function

```python
def linear_function(x):
    return x

print(linear_function(-10))  # Output: -10
print(linear_function(10))   # Output: 10
print(linear_function(1))    # Output: 1
```

## Open in Colab

You can open this notebook in Google Colab for an interactive experience:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KauaHenSilva/python_neural_network_activation_function/blob/main/main.ipynb)

## License

This project does not have a specific license. 

## Contact

For any inquiries or issues, you can reach out via the [GitHub profile](https://github.com/KauaHenSilva).

Feel free to edit and expand this README to better fit your repository's needs.
