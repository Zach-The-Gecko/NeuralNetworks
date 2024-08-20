// That last one was GARBAGE, hopefully this one will be better

import fs from "fs";

// Network: takes in 3 inputs, does some calculations, and outputs 3 outputs with the weights and biases as the parameter
// Cost:    takes in the weights and biases as inputs, outputs how BAD the function is, training data as parameters

interface TrainingData {
  inputs: number[];
  outputs: number[];
}

interface Layer {
  weights: number[][];
  biases: number[];
  layerPosition: number;
}

interface Network {
  layers: Layer[];
}

const deepLog = (obj: any) => {
  console.dir(obj, { depth: null });
};

//const createArrayOfRandomVals = (length: number) => {
//return Array.from({ length }, () => Math.random() * 2 - 1);
//};

const createArrayOfRandomVals = (length: number) => {
  return Array.from(
    { length },
    () => Math.round((Math.random() * 2 - 1) * 10) / 10
  );
};

const getTrainingData = () => {
  const irisRaw = fs.readFileSync("IRIS.csv", "utf-8");

  const irisRows = irisRaw.split("\r\n");

  const trainingData: TrainingData[] = irisRows.slice(1).map((csvRow) => {
    const inputs = csvRow
      .split(",")
      .slice(0, 4)
      .map((inputPoint) => parseFloat(inputPoint));
    const outputString = csvRow.split(",")[4];
    let outputs: number[];
    switch (outputString) {
      case "Iris-setosa":
        outputs = [1, 0, 0];
        break;
      case "Iris-versicolor":
        outputs = [0, 1, 0];
        break;
      case "Iris-virginica":
        outputs = [0, 0, 1];
        break;
      default:
        outputs = [];
    }
    return { inputs, outputs };
  });

  trainingData.pop();

  for (let i = trainingData.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [trainingData[i], trainingData[j]] = [trainingData[j], trainingData[i]];
  }

  return trainingData;
};

// const createArrayOfRandomVals = (length: number) => {
// return Array.from({ length }, (_) => {
// counter++;
// return counter / 10;
// });
// };

class Network {
  constructor(size: number[]) {
    this.layers = size.map((layerSize, layerPosition, layerSizeArr) => {
      if (layerPosition === 0) {
        return {
          biases: [],
          weights: [],
          layerPosition,
        };
      } else {
        const biases = createArrayOfRandomVals(layerSize);
        const weights = Array.from({ length: layerSize }, () =>
          createArrayOfRandomVals(layerSizeArr[layerPosition - 1])
        );
        return { biases, weights, layerPosition };
      }
    });
  }

  static activationFunction(input: number) {
    return 1 / (1 + Math.exp(-input));
  }

  getLayers() {
    return this.layers;
  }

  // This method works, but it loops over each of the 'biases'. I think
  // There is probably a better way to do this
  getLayerOutput(inputs: number[], layerPosition: number): number[] {
    if (layerPosition === 0) {
      return inputs;
    } else {
      const previousLayerOutput = this.getLayerOutput(
        inputs,
        layerPosition - 1
      );
      const output = this.layers[layerPosition].biases.map(
        (bias, biasIndex) => {
          const weightedSum = previousLayerOutput.reduce(
            (acc, prevOutput, prevOutputIndex) => {
              return (
                acc +
                prevOutput *
                  this.layers[layerPosition].weights[biasIndex][prevOutputIndex]
              );
            },
            0
          );
          return Network.activationFunction(weightedSum + bias);
        }
      );
      return output;
    }
  }

  // Test if inputs is the right length
  calc(inputs: number[]) {
    return this.getLayerOutput(inputs, this.layers.length - 1);
  }

  // Test if networkOoutputs and expectedOutputs are the same
  cost(trainingData: TrainingData[]) {
    const totalCost = trainingData.reduce((acc, dataPoint) => {
      const networkOutputs = this.calc(dataPoint.inputs);
      const expectedOutputs = dataPoint.outputs;

      let cost = 0;
      for (let i = 0; i < networkOutputs.length; i++) {
        cost += (networkOutputs[i] - expectedOutputs[i]) ** 2;
      }

      return acc + cost;
    }, 0);
    return totalCost / trainingData.length;
  }

  calcWeightGradients(trainingData: TrainingData[]) {
    const h = 0.0001;
    const previousNetworkCost = this.cost(trainingData);
    return this.layers.map((layer, layerIndex) => {
      return layer.weights.map((weights, weightsIndex) => {
        return weights.map((_weight, weightIndex) => {
          this.layers[layerIndex].weights[weightsIndex][weightIndex] += h;
          const newNetworkCost = this.cost(trainingData);
          this.layers[layerIndex].weights[weightsIndex][weightIndex] -= h;

          const weightGradient = (previousNetworkCost - newNetworkCost) / h;
          return weightGradient;
        });
      });
    });
  }

  calcBiasGradients(trainingData: TrainingData[]) {
    const h = 0.0001;
    const previousNetworkCost = this.cost(trainingData);
    return this.layers.map((layer, layerIndex) => {
      return layer.biases.map((_bias, biasIndex) => {
        this.layers[layerIndex].biases[biasIndex] += h;
        const newNetworkCost = this.cost(trainingData);
        this.layers[layerIndex].biases[biasIndex] += h;

        const biasGradient = (previousNetworkCost - newNetworkCost) / h;
        // console.log({ weightsIndex, weight, newNetworkCost, weightGradient });
        return biasGradient;
      });
    });
  }

  updateNetwork(weightGradients: number[][][], biasGradients: number[][]) {
    const learnRate = 0.1;
    weightGradients.map((layerWeightsGradient, layerWeightsGradientIndex) => {
      layerWeightsGradient.map(
        (nodeWeightsGradient, nodeWeightsGradientIndex) => {
          this.layers[layerWeightsGradientIndex].biases[
            nodeWeightsGradientIndex
          ] +=
            biasGradients[layerWeightsGradientIndex][nodeWeightsGradientIndex] *
            learnRate;
          nodeWeightsGradient.map((weightGradient, weightGradientIndex) => {
            this.layers[layerWeightsGradientIndex].weights[
              nodeWeightsGradientIndex
            ][weightGradientIndex] += weightGradient * learnRate;
          });
        }
      );
    });
  }

  trainNetwork(trainingData: TrainingData[], batchSize: number, depth: number) {
    const miniBatches = trainingData.reduce((acc, _, index) => {
      if (index % batchSize === 0) {
        acc.push(trainingData.slice(index, index + batchSize));
      }
      return acc;
    }, [] as TrainingData[][]);

    for (let i = 0; i < depth; i++) {
      const miniBatch = miniBatches[i % miniBatches.length];
      const weightsGradients = myNet.calcWeightGradients(miniBatch);

      const biasesGradients = myNet.calcBiasGradients(miniBatch);

      myNet.updateNetwork(weightsGradients, biasesGradients);
    }
  }

  testNetwork(trainingData: TrainingData[]) {
    return trainingData.map((data) => {
      const networkOutputs = this.calc(data.inputs);
      const expectedOutputs = data.outputs;

      const networkResult = networkOutputs.reduce(
        (acc, output, index) => {
          if (output > acc.highestValue) {
            return { highestValue: output, index };
          }
          return acc;
        },
        { highestValue: 0, index: 0 }
      );

      const correctResult = expectedOutputs.reduce(
        (acc, output, index) => {
          if (output > acc.highestValue) {
            return { highestValue: output, index };
          }
          return acc;
        },
        { highestValue: 0, index: 0 }
      );

      const correctlyClassified = networkResult.index === correctResult.index;

      return { networkOutputs, expectedOutputs, correctlyClassified };
    });
  }
}

const myNet = new Network([4, 4, 3, 3]);

const completeData = getTrainingData();

const splitIndex = Math.floor(completeData.length * 0.8);

const trainingData = completeData.slice(0, splitIndex);
const testData = completeData.slice(splitIndex);

myNet.trainNetwork(trainingData, 10, 100000);

const trainingTestResults = myNet.testNetwork(trainingData);
const testTestResults = myNet.testNetwork(testData);

const incorrectlyClassifiedTraining = trainingTestResults.filter((result) => {
  return !result.correctlyClassified;
});

const incorrectlyClassifiedTest = testTestResults.filter((result) => {
  return !result.correctlyClassified;
});

console.log(
  `Test classified ${parseFloat(
    (
      (incorrectlyClassifiedTraining.length * 100) /
      trainingTestResults.length
    ).toFixed(2)
  )}% of the training data incorrectly, and ${parseFloat(
    ((incorrectlyClassifiedTest.length * 100) / testTestResults.length).toFixed(
      2
    )
  )}% of the testData incorrectly.`
);

console.log("Incorrectly classified from training set:");
deepLog(incorrectlyClassifiedTraining);
console.log(
  "------------------------------------------------------------------------------------"
);
console.log(
  "Incorrectly classified from test set (data that has never seen the network):"
);
deepLog(incorrectlyClassifiedTest);

fs.writeFileSync("networkData.json", JSON.stringify(myNet.layers));
