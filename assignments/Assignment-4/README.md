# Neural Network Backpropagation with Generic Activation Functions

## Architecture
- **Input Layer:** 4 neurons
- **Hidden Layer 1:** 2 neurons  
- **Hidden Layer 2:** 4 neurons
- **Hidden Layer 3:** 2 neurons
- **Output Layer:** 2 neurons

## Notation
- $L$: Loss function
- $z^{(l)}_j$: Pre-activation for neuron $j$ in layer $l$
- $a^{(l)}_j$: Activation output for neuron $j$ in layer $l$
- $W^{(l)}_{jk}$: Weight connecting neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$
- $b^{(l)}_j$: Bias for neuron $j$ in layer $l$
- $g(z)$: Generic activation function for hidden layers
- $g'(z)$: Derivative of activation function

## Forward Pass

### Input Layer
$$a^{(0)}_k = x_k \text{ for } k=1, \dots, 4$$

### Hidden Layer 1
$$z^{(1)}_j = \sum_{k=1}^{4} W^{(1)}_{jk} a^{(0)}_k + b^{(1)}_j$$
$$a^{(1)}_j = g(z^{(1)}_j)$$

### Hidden Layer 2
$$z^{(2)}_j = \sum_{k=1}^{2} W^{(2)}_{jk} a^{(1)}_k + b^{(2)}_j$$
$$a^{(2)}_j = g(z^{(2)}_j)$$

### Hidden Layer 3
$$z^{(3)}_j = \sum_{k=1}^{4} W^{(3)}_{jk} a^{(2)}_k + b^{(3)}_j$$
$$a^{(3)}_j = g(z^{(3)}_j)$$

### Output Layer
$$z^{(4)}_j = \sum_{k=1}^{2} W^{(4)}_{jk} a^{(3)}_k + b^{(4)}_j$$
$$a^{(4)}_j = \text{softmax}(z^{(4)}_j) = \frac{e^{z^{(4)}_j}}{\sum_{p=1}^{2} e^{z^{(4)}_p}}$$

## Loss Function
$$L = -\sum_{j=1}^{2} t_j \log(a^{(4)}_j)$$

## Backpropagation

### Output Layer (L=4)
**Error Term:**
$$\delta^{(4)}_j = a^{(4)}_j - t_j$$

**Weight Gradients:**
$$\frac{\partial L}{\partial W^{(4)}_{jk}} = \delta^{(4)}_j a^{(3)}_k$$

**Bias Gradients:**
$$\frac{\partial L}{\partial b^{(4)}_j} = \delta^{(4)}_j$$

### Hidden Layer 3 (L=3)
**Error Term:**
$$\delta^{(3)}_k = \left( \sum_{j=1}^{2} W^{(4)}_{jk} \delta^{(4)}_j \right) g'(z^{(3)}_k)$$

**Weight Gradients:**
$$\frac{\partial L}{\partial W^{(3)}_{kl}} = \delta^{(3)}_k a^{(2)}_l$$

**Bias Gradients:**
$$\frac{\partial L}{\partial b^{(3)}_k} = \delta^{(3)}_k$$

### Hidden Layer 2 (L=2)
**Error Term:**
$$\delta^{(2)}_l = \left( \sum_{k=1}^{2} W^{(3)}_{kl} \delta^{(3)}_k \right) g'(z^{(2)}_l)$$

**Weight Gradients:**
$$\frac{\partial L}{\partial W^{(2)}_{lm}} = \delta^{(2)}_l a^{(1)}_m$$

**Bias Gradients:**
$$\frac{\partial L}{\partial b^{(2)}_l} = \delta^{(2)}_l$$

### Hidden Layer 1 (L=1)
**Error Term:**
$$\delta^{(1)}_m = \left( \sum_{l=1}^{4} W^{(2)}_{lm} \delta^{(2)}_l \right) g'(z^{(1)}_m)$$

**Weight Gradients:**
$$\frac{\partial L}{\partial W^{(1)}_{mn}} = \delta^{(1)}_m a^{(0)}_n$$

**Bias Gradients:**
$$\frac{\partial L}{\partial b^{(1)}_m} = \delta^{(1)}_m$$

## General Formulas

### Error Term for Hidden Layer $l$
$$\delta^{(l)}_j = \left( \sum_{p=1}^{n_{l+1}} W^{(l+1)}_{pj} \delta^{(l+1)}_p \right) g'(z^{(l)}_j)$$

### Weight Gradients
$$\frac{\partial L}{\partial W^{(l)}_{jk}} = \delta^{(l)}_j a^{(l-1)}_k$$

### Bias Gradients
$$\frac{\partial L}{\partial b^{(l)}_j} = \delta^{(l)}_j$$
