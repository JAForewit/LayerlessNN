
import LayerlessNN.*;
import TrainSet.*;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        // Create a Layerless Neural Network defined by a text file (see README.md)
        LayerlessNN net = new LayerlessNN("small.structure");

        // Create two simple training sets
        TrainSet trainSet = new TrainSet(net.getInputCount(),net.getOutputCount());
        double[] input1 = {2};
        double[] target1 = {1,0};
        double[] input2 = {-2};
        double[] target2 = {0,1};
        trainSet.addData(input1, target1);
        trainSet.addData(input2, target2);

        // Train the network on the training sets while recording run time
        long time = System.nanoTime();
        net.train(trainSet,10000,2,0.3);
        time = System.nanoTime() - time;

        // Display the results after training
        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input1)));
        System.out.println("MSE: " + net.MSE(input1, target1));
        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input2)));
        System.out.println("MSE: " + net.MSE(input2, target2));
        System.out.println("training time (ms): " + time/1000000);
    }
}


