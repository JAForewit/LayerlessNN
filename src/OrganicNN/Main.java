package OrganicNN;

import java.util.Arrays;
import TrainSet.TrainSet;

public class Main {
    public static void main(String[] args) {


        try {
            ONN net = new ONN("test.structure");

            int iterations = 10000;
            double rate = 0.3;
            double[] inputs = {2d, 3d};
            double[] targets = {0.3, 0.7, 0.1};

            net.train(inputs, targets, rate, iterations);

            System.out.println("\nResults:");
            System.out.println(Arrays.toString(net.calculateOutputs(inputs)));
            System.out.printf("MSE: %.3f", net.MSE(inputs,targets));

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}


