
import java.util.Arrays;
import OrganicDNN.*;

public class Main {
    public static void main(String[] args) {

        try {
            OrganicDNN net = new OrganicDNN("test.structure");

            int iterations = 10000;
            double rate = 0.3;
            double[] inputs = {2d, 3d};
            double[] targets = {0.1, 0.9};

            net.train(inputs, targets, rate, iterations);

            System.out.println("\nResults:");
            System.out.println(Arrays.toString(net.calculateOutputs(inputs)));
            System.out.printf("MSE: %.3f", net.MSE(inputs,targets));

        } catch (Exception e) { e.printStackTrace(); }

    }
}


