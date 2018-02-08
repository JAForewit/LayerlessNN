package com.jaforewit.onn;

import java.util.Arrays;
import java.io.FileNotFoundException;
import java.security.InvalidParameterException;

public class Main {
    public static void main(String[] args) {

        Neuron n = new Neuron(2);

        double out1 = n.sigmoid(0.3);
        double out2 = n.sigmoid(n.logit(out1) + 0.7*2d);
        double out3 = n.sigmoid(n.logit(out2) + 3*-0.2);

        System.out.println(out1);
        System.out.println(out2);
        System.out.println(out3);

        /*
        try {
            ONN net = new ONN("example3.structure");

            int iterations = 1000;
            double rate = 0.1;
            double[] inputs = {2, 5};
            double[] targets = {1, 0};

            net.train(inputs, targets, rate, iterations);
            net.feedForward(inputs);

            System.out.println("\nResults:");
            System.out.println(Arrays.toString(net.getLatestOutputs()));
            System.out.printf("MSE: %.3f", net.MSE(inputs,targets));
        }
        catch (InvalidParameterException e) {
            System.out.println("\nFailed to feed forward through the network."
                    + "\nInput values were invalid.");
            e.printStackTrace(System.out);
        }
        catch (FileNotFoundException e) {
            System.out.println("\nThe requested .structure file does not exist."
                    + "\nPlease review README.md for formatting instructions.");
            e.printStackTrace(System.out);
        }
        catch (Exception e) {
            System.out.println("\nThe given .structure file has an invalid format."
                    + "\nPlease review README.md for formatting instructions.");
            e.printStackTrace(System.out);
        }
        */
    }
}


