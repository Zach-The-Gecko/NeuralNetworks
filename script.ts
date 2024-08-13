import csv from "csv-parser";
import path from "path";
import fs from "fs";

interface Layer {
  size: number;
  position: number;
}

interface CSVRow {
  [key: string]: string;
}

interface Network {
  size: number[];
  layers: Layer[];
  data: CSVRow[];
}


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

class Layer {
  constructor(size: number, position: number) {
    this.size = size;
    this.position = position;
  }
}

class Network {
  constructor(size: number[], data: CSVRow[]) {
    this.size = size;
    this.layers = [];
    this.data = data;
    for (let i = 0; i < size.length; i++) {
      this.layers[i] = new Layer(size[i]);
    }
  }
  getNetworkOutput(inputs: number[]){
    
  }
}

const myNetwork = new Network([2, 3, 3], parseCSV("./IRIS.csv"));
myNetwork.calculateCost();
