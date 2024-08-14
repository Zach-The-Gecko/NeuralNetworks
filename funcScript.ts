// I could possibly remove the 'inputSize' attribute from the layerInfo object
import fs from "fs";
import path from "path";

type activationFunction = (input: number) => number;

interface CSVRow {
  [key: string]: string;
}

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

const calculateWeightGradientForSingleInput = (
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

const calculateBiasGradientForSingleInput = (
  network: NetworkInfo,
  inputs: number[],
  expectedOutputs: number[]
) => {
  return network.layers.map((layer, layerPosition) => {
    return layer.neurons.map((neuron, neuronPosition) => {
      const previousNetworkOutputs = testInputsOnNetwork(inputs, network);
      const previousNetworkCost = calculateCost(
        previousNetworkOutputs,
        expectedOutputs
      );

      network.layers[layerPosition].neurons[neuronPosition].bias += 0.00001;
      const newNetworkOutputs = testInputsOnNetwork(inputs, network);
      const newNetworkCost = calculateCost(newNetworkOutputs, expectedOutputs);

      return (newNetworkCost - previousNetworkCost) / 0.00001;
    });
  });
};

const updateNetwork = (
  network: NetworkInfo,
  weightGradients: number[][][],
  biasGradients: number[][]
) => {
  const newNetwork: NetworkInfo = { layers: [], activationFunction: () => 0 };
  newNetwork.layers = structuredClone(network.layers);
  newNetwork.activationFunction = network.activationFunction;
  newNetwork.layers = structuredClone(network.layers);

  const learnRate = 5;
  weightGradients.map((layerWeightGradients, layerWeightGradientsPosition) => {
    layerWeightGradients.map(
      (neuronWeightGradients, neuronWeightGradientsPosition) => {
        neuronWeightGradients.map((weightGradient, weightGradientPosition) => {
          newNetwork.layers[layerWeightGradientsPosition].neurons[
            neuronWeightGradientsPosition
          ].weights[weightGradientPosition] -= weightGradient * learnRate;
        });
      }
    );
  });

  biasGradients.map((layerBiasGradients, layerBiasGradientsPosition) => {
    layerBiasGradients.map((neuronBiasGradient, neuronBiasPosition) => {
      newNetwork.layers[layerBiasGradientsPosition].neurons[
        neuronBiasPosition
      ].bias -= neuronBiasGradient * learnRate;
    });
  });

  return newNetwork;
};

const trainOnSingleInputTEST = (
  network: NetworkInfo,
  depth: number,
  inputs: number[],
  expectedOutputs: number[]
) => {
  if (depth === 0) {
    return network;
  }
  const weightGradients = calculateWeightGradientForSingleInput(
    network,
    inputs,
    expectedOutputs
  );
  const biasGradients = calculateBiasGradientForSingleInput(
    network,
    inputs,
    expectedOutputs
  );

  const newNetwork = updateNetwork(network, weightGradients, biasGradients);
  return trainOnSingleInputTEST(newNetwork, depth - 1, inputs, expectedOutputs);
};

const parseCSV = (filePath: string): CSVRow[] => {
  const fileContent = fs.readFileSync(path.resolve(filePath), "utf-8");
  const lines = fileContent
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const headers = lines[0].split(",");
  const result: CSVRow[] = [];

  for (let i = 1; i < lines.length; i++) {
    const currentLine = lines[i].split(",");
    const row: CSVRow = {};

    headers.forEach((header, index) => {
      row[header] = currentLine[index];
    });

    result.push(row);
  }

  return result;
};

const mapSpeciesToExpectedOutput = (species: string) => {
  switch (species) {
    case "Iris-setosa":
      return [1, 0, 0];
    case "Iris-versicolor":
      return [0, 1, 0];
    case "Iris-virginica":
      return [0, 0, 1];
  }
  return [0, 0, 0];
};

// WORK ON THIS ONE
const trainNetworkOneIteration = (
  network: NetworkInfo,
  trainingData: number[][]
) => {
  const totalCost = trainingData.reduce((acc, [inputs, outputs]) => {
    return acc;
  }, 0);
};

const formattedData = parseCSV("IRIS.csv").map(
  ({ sepal_length, sepal_width, petal_length, petal_width, species }) => {
    return [
      [
        parseFloat(sepal_length),
        parseFloat(sepal_width),
        parseFloat(petal_length),
        parseFloat(petal_width),
      ],
      mapSpeciesToExpectedOutput(species),
    ];
  }
);

console.log(formattedData);

// const network = initializeNetwork([2, 2], sigmoid);
