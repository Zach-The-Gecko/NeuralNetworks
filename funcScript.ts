// I could possibly remove the 'inputSize' attribute from the layerInfo object

type activationFunction = (input: number) => number;

interface NeuronInfo {
  bias: number;
  weights: number[];
}

interface layerInfo {
  inputSize: number;
  inputLayer: Boolean;
  neurons: NeuronInfo[];
}

interface NetworkInfo {
  layers: layerInfo[];
  activationFunction: activationFunction;
}

const deepLog = (obj: any) => {
  console.dir(obj, { depth: null });
};

const ReLU: activationFunction = (input) => {
  return input > 0 ? input : 0;
};

const sigmoid: activationFunction = (input) => {
  return 1 / (1 + Math.exp(-input));
};

const initializeLayer = (
  inputSize: number,
  outputSize: number,
  layerPosition: number
): layerInfo => {
  const inputLayer = layerPosition === 0;
  const emptyNeurons = Array.from(Array(outputSize), () => 0);
  const initializedNeurons: NeuronInfo[] = emptyNeurons.map(() => {
    return {
      bias: (Math.random() - 1) * 2,
      weights: Array.from(Array(inputSize), () => (Math.random() - 1) * 2),
    };
  });
  return { neurons: initializedNeurons, inputSize, inputLayer };
};

const initializeNetwork = (
  layerSizes: number[],
  activationFunction: activationFunction
): NetworkInfo => {
  const initializedLayers: layerInfo[] = layerSizes.map(
    (layerSize, layerPosition) => {
      const prevLayerSize =
        layerPosition === 0 ? 0 : layerSizes[layerPosition - 1];
      return initializeLayer(prevLayerSize, layerSize, layerPosition);
    }
  );
  return { layers: initializedLayers, activationFunction };
};

const getLayerOutput = (
  inputs: number[],
  network: NetworkInfo,
  layer: number
): number[] => {
  if (layer === 0) {
    return inputs;
  }

  const previousLayerOutput: number[] = getLayerOutput(
    inputs,
    network,
    layer - 1
  );
  const currentLayerOutput: number[] = network.layers[layer].neurons.map(
    ({ bias, weights }) => {
      const weightedSum = previousLayerOutput.reduce(
        (acc: number, prevNeuronOutput, prevNeuronPosition: number) => {
          return acc + prevNeuronOutput * weights[prevNeuronPosition];
        },
        0
      );
      return network.activationFunction(weightedSum + bias);
    }
  );
  return currentLayerOutput;
};

const testInputsOnNetwork = (inputs: number[], network: NetworkInfo) => {
  if (inputs.length !== network.layers[0].neurons.length) {
    throw Error("Inputs does not match network");
  } else {
    return getLayerOutput([1, 1], network, network.layers.length - 1);
  }
};

const calculateCost = (outputs: number[], expectedOutputs: number[]) => {
  const cost = outputs.reduce((acc, output, outputIndex) => {
    return acc + (output - expectedOutputs[outputIndex]) ** 2;
  }, 0);
  return cost;
};

const calculateGradientForSingleInput = (
  network: NetworkInfo,
  inputs: number[],
  expectedOutputs: number[]
) => {
  return network.layers.map((layer, layerPosition) => {
    return layer.neurons.map((neuron, neuronPosition) => {
      return neuron.weights.map((weight, weightPosition) => {
        const previousNetworkOutputs = testInputsOnNetwork(inputs, network);
        const previousNetworkCost = calculateCost(
          previousNetworkOutputs,
          expectedOutputs
        );

        network.layers[layerPosition].neurons[neuronPosition].weights[
          weightPosition
        ] += 0.00001;
        const newNetworkOutputs = testInputsOnNetwork(inputs, network);
        const newNetworkCost = calculateCost(
          newNetworkOutputs,
          expectedOutputs
        );

        return (newNetworkCost - previousNetworkCost) / 0.00001;
      });
    });
  });
};

const updateNetwork = (network: NetworkInfo, weights: number[][][]) => {
  const learnRate = 5;
  weights.map((layerWeights, layerWeightsPosition) => {
    layerWeights.map((neuronWeights, neuronWeightsPosition) => {
      neuronWeights.map((weight, weightPosition) => {
        network.layers[layerWeightsPosition].neurons[
          neuronWeightsPosition
        ].weights[weightPosition] -= weight * learnRate;
      });
    });
  });
  return network;
};

// console.log(testInputsOnNetwork([1, 1], network));

// console.log(calculateCost([0.3, 0.2, 0.6], [0, 0, 1]));

// deepLog(network);
// console.log(testInputsOnNetwork([2, 3], network));

const network = initializeNetwork([2, 3, 3, 2], sigmoid);
const d1 = calculateGradientForSingleInput(network, [2, 2], [1, 0]);

const network2 = updateNetwork(network, d1);
const d2 = calculateGradientForSingleInput(network2, [2, 2], [1, 0]);

// I'm close, but I also have to update gradient for biases, not just weights
