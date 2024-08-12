interface Layer {
  outputs: number;
  layerPosition: number;
}

interface Network {
  layerSizes: number[];
  layers: Layer[];
}

class Layer {
  constructor(inputs: number, outputs: number, layerPosition: number) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.layerPosition = layerPosition;
    console.log("New layer created: ", sizeOfLayer, " and position: ", layerPosition);
  }
}

class Network {
  constructor(layerSizes: number[]) {
    this.layerSizes = layerSizes;
    this.layers = [];
    for (let i: number = 0; i < layerSizes.length; i++) {
      this.layers[i] = new Layer(layerSizes[i], i);
    }
  }
}

const myNetwork = new Network([1, 1, 1]);
