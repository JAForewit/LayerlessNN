package com.jaforewit.onn;

public class Main {
    public static void main(String[] args) {

        try {
            ONN net = new ONN("example.structure");

            try {
                double[] inputs = {1,2,3};
                double[] outputs = net.feedForward(inputs);

                System.out.println("Feed forward results:");
                for (double output : outputs) System.out.println("\t" + output);
            }
            catch (Exception e) {
                System.out.println("\nFailed to feed forward the network.");
                e.printStackTrace(System.out);
            }

        } catch (Exception e) {}

    }
}


