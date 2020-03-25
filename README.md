# MultiLayer NeuralNetwork
Javascript library for developing some efficient and easy to use neural networks.

### Installation
You could import this library using a script tag:
```html
<script src="neuralnet.js" charset="utf-8"></script>
```
if you wan't to use this library with node, use:
```javascript
const {NeuralNetwork} = require("path/to/neuralnet.js");
```

#### Usage
```javascript
let neuralNetwork = new NeuralNetwork([2, 3, 2, 1])
neuralNetwork.setActivation("tanh") // default is sigmoid
neuralNetwork.setLearningRate(0.05) // default is 0.01
neuralNetwork.feed([0.9, 0.1]) // returns an array of all outputs (1)
neuralNetwork.train([0, 1], [0]) // returns undefined
```
If you have your own activation function, you could provide it using:
```javascript
neuralNetwork.setActivation("custom", myFunction, myDerivativeFunction)
```
