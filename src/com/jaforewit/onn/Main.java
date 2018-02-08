package com.jaforewit.onn;

import java.util.Arrays;
import java.io.FileNotFoundException;
import java.security.InvalidParameterException;

public class Main {
    public static void main(String[] args) {


        try {
            ONN net = new ONN("test.structure", "test.weights");

            int iterations = 1005;
            double rate = 0.3;
            double[] inputs = {2d, 3d};
            double[] targets = {0.5};

            net.train(inputs, targets, rate, iterations);
            net.feedForward(inputs);
            net.printNet();


            System.out.println("\nResults:");
            System.out.println(Arrays.toString(net.getLatestOutputs()));
            System.out.printf("MSE: %.3f", net.MSE(inputs,targets));
        }
        catch (Exception e) { e.printStackTrace(System.out); }

    }
}


