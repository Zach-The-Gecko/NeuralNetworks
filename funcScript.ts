// I could possibly remove the 'inputSize' attribute from the layerInfo object
import exp from "constants";
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
    return getLayerOutput(inputs, network, network.layers.length - 1);
  }
};

const calculateCost = (outputs: number[], expectedOutputs: number[]) => {
  const cost = outputs.reduce((acc, output, outputIndex) => {
    return acc + (output - expectedOutputs[outputIndex]) ** 2;
  }, 0);
  return cost;
};

// Problem has GOTTA be in here... right?
const calculateAverageCost = (
  outputsList: number[][],
  expectedOutputsList: number[][]
) => {
  // console.log("----------------------------------------------------");
  // console.log({ outputsList, expectedOutputsList });
  const totalCost = outputsList.reduce((acc, outputs, outputsPosition) => {
    return (acc += calculateCost(
      outputs,
      expectedOutputsList[outputsPosition]
    ));
  }, 0);
  // console.log(totalCost);
  return totalCost / outputsList.length;
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

        // network.layers[layerPosition].neurons[neuronPosition].weights[
        // weightPosition
        // ] -= 0.00001;

        return (newNetworkCost - previousNetworkCost) / 0.00001;
      });
    });
  });
};

const calculateWeightGradient = (
  network: NetworkInfo,
  inputsList: number[][],
  expectedOutputsList: number[][]
) => {
  const networkOutputsList = inputsList.map((inputs) => {
    return testInputsOnNetwork(inputs, network);
  });
  const prevNetworkAvgCost = calculateAverageCost(
    networkOutputsList,
    expectedOutputsList
  );
  return network.layers.map((layer, layerPosition) => {
    return layer.neurons.map((neuron, neuronPosition) => {
      return neuron.weights.map((weight, weightPosition) => {
        network.layers[layerPosition].neurons[neuronPosition].weights[
          weightPosition
        ] += 0.00001;
        const newNetworkOutputsList = inputsList.map((inputs) => {
          // console.log(inputs);
          const output = testInputsOnNetwork(inputs, network);
          // console.log(output);
          return output;
        });
        const newNetworkCost = calculateAverageCost(
          newNetworkOutputsList,
          expectedOutputsList
        );
        // console.log({
        // inputsList,
        // networkOutputsList,
        // expectedOutputsList,
        // networkDeriv: (newNetworkCost - prevNetworkAvgCost) / 0.00001,
        // });
        return (newNetworkCost - prevNetworkAvgCost) / 0.00001;
        // Add small value, test inputs, calculate new avg cost, return the new cost - old cost / small value
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

const calculateBiasGradient = (
  network: NetworkInfo,
  inputsList: number[][],
  expectedOutputsList: number[][]
) => {
  const networkOutputsList = inputsList.map((inputs) => {
    return testInputsOnNetwork(inputs, network);
  });
  const prevNetworkAvgCost = calculateAverageCost(
    networkOutputsList,
    expectedOutputsList
  );
  return network.layers.map((layer, layerPosition) => {
    return layer.neurons.map((neuron, neuronPosition) => {
      network.layers[layerPosition].neurons[neuronPosition].bias += 0.00001;
      const newNetworkOutputsList = inputsList.map((inputs) => {
        return testInputsOnNetwork(inputs, network);
      });
      const newNetworkCost = calculateAverageCost(
        newNetworkOutputsList,
        expectedOutputsList
      );
      return (newNetworkCost - prevNetworkAvgCost) / 0.00001;
    });
  });
};
const updateNetwork = (
  network: NetworkInfo,
  weightGradients: number[][][],
  biasGradients: number[][]
) => {
  const newNetwork: NetworkInfo = {
    activationFunction: network.activationFunction,
    layers: structuredClone(network.layers),
  };

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
  trainingData: number[][][]
) => {
  const inputs = trainingData.map((trainingData) => trainingData[0]);
  const outputs = trainingData.map((trainingData) => trainingData[1]);
  const weightGradient = calculateWeightGradient(network, inputs, outputs);
  const biasGradient = calculateBiasGradient(network, inputs, outputs);
  return updateNetwork(network, weightGradient, biasGradient);
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

// const network = initializeNetwork([2, 6, 6, 6, 2], sigmoid);

// const newNet = trainOnSingleInputTEST(network, 5000, [3, 3], [0, 1]);

// console.log(testInputsOnNetwork([2, 2], network));
// console.log(testInputsOnNetwork([2, 2], newNet));

const irisNetwork = initializeNetwork([4, 3, 3], sigmoid);

let bestNetwork: NetworkInfo = { layers: [], activationFunction: () => 0 };
bestNetwork.layers = structuredClone(irisNetwork.layers);
bestNetwork.activationFunction = irisNetwork.activationFunction;

// for (let i = 0; i <= 1000; i++) {
// bestNetwork.layers = structuredClone(
// trainNetworkOneIteration(bestNetwork, formattedData).layers
// );
// }

const iterable = Array.from(Array(1000), () => 0);

const realBestNetwork = iterable.reduce((acc) => {
  return trainNetworkOneIteration(acc, formattedData);
}, bestNetwork);

// Its getting the best result for the AVERAGE of the outputs,
// so because each output has a 1/3 chance of being the right
// one, it trains the network to 1/3 for each

const randomNetworkResult = testInputsOnNetwork(
  [5.1, 3.5, 1.4, 0.2],
  irisNetwork
);
const trainedNetworkResult = testInputsOnNetwork(
  [5.1, 3.5, 1.4, 0.2],
  realBestNetwork
);

console.log(randomNetworkResult);
console.log(trainedNetworkResult);
console.log(testInputsOnNetwork([4.9, 3, 1.4, 0.2], bestNetwork));
console.log(testInputsOnNetwork([4.7, 3.2, 1.3, 0.2], bestNetwork));
console.log(testInputsOnNetwork([4.6, 3.1, 1.5, 0.2], bestNetwork));

// 5.1,3.5,1.4,0.2,Iris-setosa
// 4.9,3,1.4,0.2,Iris-setosa
// 4.7,3.2,1.3,0.2,Iris-setosa
// 4.6,3.1,1.5,0.2,Iris-setosa
// 5,3.6,1.4,0.2,Iris-setosa
// 5.4,3.9,1.7,0.4,Iris-setosa
// 4.6,3.4,1.4,0.3,Iris-setosa
// 5,3.4,1.5,0.2,Iris-setosa
// 4.4,2.9,1.4,0.2,Iris-setosa
// 4.9,3.1,1.5,0.1,Iris-setosa
// 5.4,3.7,1.5,0.2,Iris-setosa
// 4.8,3.4,1.6,0.2,Iris-setosa
// 4.8,3,1.4,0.1,Iris-setosa
// 4.3,3,1.1,0.1,Iris-setosa
