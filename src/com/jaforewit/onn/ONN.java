package com.jaforewit.onn;


//
// n refers to neuron
//
//

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class ONN {

    private int timeStep = 0;
    private int inputCount;
    private int outputCount;

    private Neuron neurons[];

    public ONN(String fileName) {

        try {
            /*
            STILL NEEDED: check for valid .structure file
             */
            System.out.println("Accepting input from " + fileName + ":");

            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            // expecting [input neuron count] [hidden neuron count] [output neuron count]
            int[] array = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

            inputCount =  array[0];
            outputCount = array[1];
            neurons = new Neuron[inputCount + outputCount + array[2]];

            System.out.println(array[0] + " " +  array[1]  + " " + array[2]);

            // initialize neurons
            for (int i=0; i<neurons.length; i++) neurons[i] = new Neuron();

            // set inputs for each hidden and output neuron
            for (int i = inputCount; i < neurons.length; i++) {
                int[] nextInputs = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();

                for (int j=0; j<nextInputs.length; j++) {
                    neurons[i].addInput(neurons[nextInputs[j]]);

                    System.out.print(nextInputs[j] + " ");
                }
                System.out.println();
            }

            System.out.println("Success! " + neurons.length + " neurons, "
                    + inputCount + " inputs, " + outputCount + " ouputs");

        } catch (Exception e) {
            e.printStackTrace();
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
