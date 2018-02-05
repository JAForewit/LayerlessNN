package com.jaforewit.onn;


/*
Organic Neural Network - ONN Class
This class accepts .structure files to define the network.
Please review README.md for more information on .structure files.

TODO: create a file to store weights and bias's for loading networks
TODO: make a training function and error calculation
 */

import java.io.*;
import java.security.InvalidParameterException;
import java.util.Arrays;

public class ONN {
    private final double MIN_BIAS = -0.7;
    private final double MAX_BIAS = 0.5;
    private final double MIN_WEIGHT = -1;
    private final double MAX_WEIGHT = 1;

    //private int timeStep = 0;   // the current time step
    private int inputCount;     // number of input neurons
    private int outputCount;    // number of output neurons
    private int neuronCount;
    private Neuron neurons[];   // array holding all neurons (including hidden neurons)


    public ONN(String filename) throws Exception {
        try {
            // reading .structure file
            BufferedReader reader = new BufferedReader(new FileReader(filename));

            System.out.println("Accepting input from " + filename + ":");


            // reading network critical definitions
            int[] array = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

            inputCount =  array[0];
            outputCount = array[1];
            neuronCount = inputCount + outputCount + array[2];
            neurons = new Neuron[neuronCount];

            System.out.println(array[0] + " " +  array[1]  + " " + array[2]);


            // initializing neurons with a random bias
            double randBias = (Math.random() * (MAX_BIAS - MIN_BIAS)) + MIN_BIAS;
            for (int i=0; i<neurons.length; i++) neurons[i] = new Neuron(randBias);


            // setting inputs and random weights for each hidden and output neuron
            double randWeight;
            for (int i = inputCount; i < neurons.length; i++) {
                int[] nextInputs = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

                for (int input : nextInputs) {
                    randWeight = (Math.random() * (MAX_WEIGHT - MIN_WEIGHT)) + MIN_WEIGHT;
                    neurons[i].addInput(neurons[input],randWeight);

                    System.out.print(input + " ");
                }
                System.out.println();
            }


            // verifying the file has ended
            if (reader.readLine() != null) throw new Exception();

            System.out.println("Success! " + neurons.length + " neurons, "
                    + inputCount + " inputs, " + outputCount + " ouputs");
        }
        catch (FileNotFoundException e) {
            System.out.println("\nThe requested .structure file does not exist."
                    + "Please review README.md for formatting instructions.");
            e.printStackTrace(System.out);
            throw e;
        }
        catch (Exception e) {
            System.out.println("\nThe given .structure file has an invalid format."
                    + "\nPlease review README.md for formatting instructions.");
            e.printStackTrace(System.out);
            throw e;
        }
    }


    public double[] feedForward(double[] inputs) {
        if (inputs.length != inputCount) throw new InvalidParameterException();

        // assigning input values
        for (int i=0; i<inputCount; i++) neurons[i].setOutput(inputs[i]);

        double[] outputs = new double[outputCount];
        for (int i=0; i<outputCount; i++) {
            outputs[i] = neurons[neuronCount - i - 1].getOutput();
        }

        return outputs;
    }



    private boolean backprop() {
        return true;
    }
}
