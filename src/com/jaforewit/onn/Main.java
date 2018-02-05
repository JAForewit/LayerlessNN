package com.jaforewit.onn;

public class Main {
    public static void main(String[] args) {
        ONN net = new ONN("example.structure");
;
        double[] inputs = {1,2,3};
        double[] outputs = net.feedForward(inputs);

        System.out.println("Results:");
        for (double output : outputs) {
            System.out.println("\t" + output);
        }
    }
}


