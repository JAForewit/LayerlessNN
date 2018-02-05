package com.jaforewit.onn;


/*
Organic Neural Network - ONN Class
 */

import java.io.*;
import java.util.Arrays;

public class ONN {

    private int timeStep = 0;   // the current timestep
    private int inputCount;     // number of input neurons
    private int outputCount;    // number of output neurons
    private Neuron neurons[];   // array holding all neurons (including hidden neurons)


    public ONN(String filename) {
        try {
            // reading .structure file
            BufferedReader reader = new BufferedReader(new FileReader(filename));

            System.out.println("Accepting input from " + filename + ":");


            // reading network critical definitions
            int[] array = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

            inputCount =  array[0];
            outputCount = array[1];
            neurons = new Neuron[inputCount + outputCount + array[2]];

            System.out.println(array[0] + " " +  array[1]  + " " + array[2]);


            // initializing neurons based on critical definitions
            for (int i=0; i<neurons.length; i++) neurons[i] = new Neuron();


            // setting inputs for each hidden and output neuron
            for (int i = inputCount; i < neurons.length; i++) {
                int[] nextInputs = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

                for (int input : nextInputs) {
                    neurons[i].addInput(neurons[input]);

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
        }
        catch (Exception e) {
            System.out.println("\nThe given .structure file has an invalid format."
                    + "\nPlease review README.md for formatting instructions.");
            e.printStackTrace(System.out);
        }
    }


    private boolean feed() {
        return true;
    }


    private boolean backprop() {
        return true;
    }


    private void sigmoid() {

    }
}
