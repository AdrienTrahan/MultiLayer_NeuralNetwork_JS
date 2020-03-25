
class NeuralNetwork{
  constructor(shape){
    this.learningRate = 0.1;
    this.shape = shape;
    this.rows = [];
    this.activation = this.sigmoid;
    this.dactivation = this.dsigmoid;
    for (var i = 0; i < this.shape.length; i++){
      let row = [];
      for (var j = 0; j < this.shape[i]; j++){
        let object = {
          bias: 0,
          weights: []
        }
        if (i != 0){
          object.bias = Math.random() * 2 - 1;
          for (var k = 0; k < this.rows[i - 1].length; k++){
            let weight = Math.random() * 2 - 1;
            object.weights.push(weight);
          }
        }
        row.push(object);
      }
      this.rows.push(row);
    }
  }

  feed(input, saveActivation = false){
    // prepare to save activations
    let activations = [input];
    // exit if shape is invalid
    if (this.rows.length <= 1){return}
    // verify if input matches shape
    if (input.length == this.rows[0].length){
      // go through each row (feed forward)
      for (var i = 1; i < this.rows.length; i++){
        var newInput = [];
        for (var j = 0; j < this.rows[i].length; j++){
          var activation = 0;
          // sum weights (weights * activation)
          for (var w = 0; w < input.length; w++){
            activation += input[w] * this.rows[i][j].weights[w];
          }
          // add bias
          activation += this.rows[i][j].bias;
          // pass result into specified activation function
          activation = this.activation(activation);
          newInput.push(activation);
        }
        // save activation
        if (saveActivation){
          activations.push(newInput);
        }
        // prepare for next iteration (next row)
        input = newInput;
      }
    }
    // return expected output
    if (saveActivation){
      return {output: input, saved: activations};
    }else{
      return input;
    }
  }

  train(input, target){
    // exit if shape is invalid
    if (this.rows.length <= 1){return}
    // verify if input and target match shape
    if (input.length == this.rows[0].length && target.length == this.rows[this.rows.length - 1].length){
      // make history of activations available
      let result = this.feed(input, true);
      let output = result.output;
      let saved = result.saved;
      // calculate errors and gradients+deltas for output layer
      // error = target - output
      let errors = target.map((x, y) => x - output[y]);
      // gradient = dactivation(output) * error * learningRate
      let gradient = output.map(x => this.dactivation(x));
      gradient = gradient.map((x, y) => x * errors[y]);
      gradient = gradient.map(x => x * this.learningRate)

      for (var i = saved.length-1; i >= 0; i--){
        // do adjustements for each biases and weights in specific row
        for (var j = 0; j < this.rows[i].length; j++){
          this.rows[i][j].bias += gradient[j];
          for (var w = 0; w < this.rows[i][j].weights.length; w++){
            // delta = gradient * lastActivation
            let deltaWeight = gradient[j] * saved[i-1][w];
            this.rows[i][j].weights[w] = this.rows[i][j].weights[w] + deltaWeight;
          }
        }

        // generate error for next iteration (previous row)
        let newErrors = [];
        if (i != 0){
          for (var j = 0; j < this.rows[i - 1].length; j++){
            let error = 0;
            for (var k = 0; k < this.rows[i].length; k++){
              // add error for one neuron
              // error = sum(weight * error)
              error += this.rows[i][k].weights[j] * errors[k];
            }
            newErrors.push(error);
          }
          errors = newErrors;
          // generate gradient for next iteration (previous row)
          // gradient = dactivation(output) * error * learningRate
          gradient = saved[i-1].map(x => this.dactivation(x));
          gradient = gradient.map((x, y) => x * errors[y]);
          gradient = gradient.map(x => x * this.learningRate)
        }
      }
    }
  }

  setActivation(name, activation, dactivation){
    if (name == "sigmoid"){
      this.activation = sigmoid;
      this.dactivation = dsigmoid;
    }else if (name == "tanh"){
      this.activation = tanh;
      this.dactivation = dtanh;
    }else if (name == "relu"){
      this.activation = relu;
      this.dactivation = drelu;
    }else if (name == "custom" && activation && dactivation){
      this.activation = activation;
      this.dactivation = dactivation;
    }
  }

  setLearningRate(rate){
    this.learningRate = rate;
  }


  relu(t) {
    return (t < 0) ? 0 : t
  }

  drelu(t){
    return (t < 0) ? 0 : 1
  }
  tanh(t) {
    return Math.tanh(t);
  }

  dtanh(t){
    return 1 - (t * t)
  }

  sigmoid(t) {
    return 1/(1+Math.exp(-t));
  }

  dsigmoid(t){
    return t * (1-t)
  }
}
