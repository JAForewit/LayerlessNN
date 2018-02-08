package com.jaforewit.onn;

import java.util.Arrays;
import java.io.FileNotFoundException;
import java.security.InvalidParameterException;

public class Main {
    public static void main(String[] args) {


        try {
            ONN net = new ONN("test.structure", "test.weights");

            int iterations = 5;
            double rate = 0.3;
            double[] inputs = {2d, 3d};
            double[] targets = {0.5};

            //net.train(inputs, targets, rate, iterations);
            net.feedForward(inputs);
            net.printNet();


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

    }
}


