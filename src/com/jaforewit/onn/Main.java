package com.jaforewit.onn;

import java.io.FileNotFoundException;
import java.security.InvalidParameterException;

public class Main {
    public static void main(String[] args) {

        try {
            ONN net = new ONN("example2.structure");
            double[] inputs = {1, 2, 3};
            double[] outputs = net.feedForward(inputs);

            System.out.println("Feed forward results:");
            for (double output : outputs) System.out.println("\t" + output);
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


