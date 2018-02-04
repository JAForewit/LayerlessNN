package com.jaforewit.onn;


//
// n refers to neuron
//
//

import java.util.Scanner;
import java.io.File;

public class ONN {


    private Neuron inputs[];   // input neurons
    private Neuron neurons[];  // hidden neurons
    private Neuron outputs[];  // output neurons

    public ONN(String fileName) {

        try {
            Scanner scanner = new Scanner(new File(fileName));

            inputs = new Neuron[scanner.nextInt()];
            neurons = new Neuron[scanner.nextInt()];
            outputs = new Neuron[scanner.nextInt()];

            System.out.println("Inputs: \t" + inputs.length);
            System.out.println("Neurons:\t" + neurons.length);
            System.out.println("Outputs:\t" + outputs.length);

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
