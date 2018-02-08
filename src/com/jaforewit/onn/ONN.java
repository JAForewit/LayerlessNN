package com.jaforewit.onn;


/*
Organic Neural Network - ONN Class
This class accepts .structure files to define the network.
Please review README.md for more information on .structure files.

TODO: create a file to store weights and bias's for loading networks
 */

import java.io.*;
import java.security.InvalidParameterException;
import java.util.Arrays;


public class ONN {
    private final double MIN_BIAS = -0.3;
    private final double MAX_BIAS = 0.3;
    private final double MIN_WEIGHT = -1.0;
    private final double MAX_WEIGHT = 1.0;

    private int inputCount;     // number of input neurons
    private int outputCount;    // number of output neurons
    private int neuronCount;    // total number of neurons
    private Neuron neurons[];   // array holding all neurons (including hidden neurons)
    private double[] latestOutputs;


    public ONN(String structureFile, String weightsFile) throws Exception{
        BufferedReader structure = new BufferedReader(new FileReader(structureFile));
        BufferedReader weights = new BufferedReader(new FileReader(weightsFile));

        System.out.println("Accepting input from " + structureFile + ":");

        // reading network critical definitions
        int[] array = Arrays.stream(structure.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

        inputCount = array[0];
        outputCount = array[1];
        neuronCount = inputCount + outputCount + array[2];
        neurons = new Neuron[neuronCount];
        latestOutputs = new double[outputCount];

        System.out.println(array[0] + " " + array[1] + " " + array[2]);

        // initializing neurons with a random bias
        double randBias = 0.3; //(Math.random() * (MAX_BIAS - MIN_BIAS)) + MIN_BIAS;
        for (int i = 0; i < neuronCount; i++) neurons[i] = new Neuron(randBias);

        // setting outputs and random weights for each hidden and output neuron
        for (int i = 0; i < neuronCount- outputCount; i++) {
            int[] nextLine = Arrays.stream(structure.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();
            double[] nextLineWeights = Arrays.stream(weights.readLine().split("\\s")).mapToDouble(Double::parseDouble).toArray();

            for (int j=0; j<nextLine.length; j++) {
                neurons[i].addOutputAxon(neurons[nextLine[j]], nextLineWeights[j]);

                System.out.print(nextLine[j] + " ");
            }
            System.out.println();
        }

        // verifying the file has ended
        if (structure.readLine() != null) throw new Exception();
        if (weights.readLine() != null) throw new Exception();

        structure.close();
        weights.close();

        System.out.println("Success! " + neuronCount + " neurons, "
                + inputCount + " inputs, " + outputCount + " outputs");
    }


    public ONN(String filename) throws Exception {
        // loading .structure file
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        System.out.println("Accepting input from " + filename + ":");

        // reading network critical definitions
        int[] array = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

        inputCount = array[0];
        outputCount = array[1];
        neuronCount = inputCount + outputCount + array[2];
        neurons = new Neuron[neuronCount];
        latestOutputs = new double[outputCount];

        System.out.println(array[0] + " " + array[1] + " " + array[2]);

        // initializing neurons with a random bias
        double randBias = (Math.random() * (MAX_BIAS - MIN_BIAS)) + MIN_BIAS;
        for (int i = 0; i < neuronCount; i++) neurons[i] = new Neuron(randBias);

        // setting outputs and random weights for each hidden and output neuron
        double randWeight;
        for (int i = 0; i < neuronCount- outputCount; i++) {
            int[] nextLine = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

            for (int output : nextLine) {
                randWeight = (Math.random() * (MAX_WEIGHT - MIN_WEIGHT)) + MIN_WEIGHT;
                neurons[i].addOutputAxon(neurons[output], randWeight);

                System.out.print(output + " ");
            }
            System.out.println();
        }

        // verifying the file has ended
        if (reader.readLine() != null) throw new Exception();

        System.out.println("Success! " + neuronCount + " neurons, "
                + inputCount + " inputs, " + outputCount + " outputs");
    }






    public double[] getLatestOutputs() {
        for (int i=0; i<outputCount; i++) latestOutputs[i] = neurons[neuronCount-i-1].getOutput();
        return latestOutputs;
    }

    public void feedForward(double[] inputs) {
        if (inputs.length != inputCount) throw new InvalidParameterException();
        for (int i=0; i<inputCount; i++) neurons[i].feedForward(inputs[i]);
    }


    public void backpropagation(double[] inputs, double[] targets, double rate) {
        if (inputs.length != inputCount || targets.length != outputCount)
            throw new InvalidParameterException();

        feedForward(inputs);

        for (int i=0; i<outputCount; i++) { neurons[i].backpropagate(targets[i], rate); }
    }


    // Mean Squared Error
    public double MSE (double[] inputs, double[] targets) {
        if (inputs.length != inputCount || targets.length != outputCount)
            throw new InvalidParameterException();

        feedForward(inputs);

        double sum = 0;
        for (int i=0; i<outputCount; i++)
            sum += (targets[i] - getLatestOutputs()[i]) * (targets[i] - getLatestOutputs()[i]);

        return sum / (2d * outputCount);
    }

    public void train (double[] inputs, double[] targets, double rate, int iterations) {
        if (inputs.length != inputCount || targets.length != outputCount)
            throw new InvalidParameterException();

        for (int i = 0; i < iterations; i++) backpropagation(inputs,targets,rate);
    }
}