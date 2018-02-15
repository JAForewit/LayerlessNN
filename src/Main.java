
import LayerlessNN.*;
import TrainSet.*;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        LayerlessNN net = new LayerlessNN("small.structure");

        TrainSet trainSet = new TrainSet(net.getInputCount(),net.getOutputCount());
        double[] input1 = {10,-5,2};
        double[] target1 = {1,0};
        double[] input2 = {6,-4,5};
        double[] target2 = {0,1};
        trainSet.addData(input1, target1);
        trainSet.addData(input2, target2);

        long time = System.nanoTime();
        net.train(trainSet,10000,2,0.3);
        time = System.nanoTime() - time;

        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input1)));
        System.out.println("MSE: " + net.MSE(input1, target1));
        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input2)));
        System.out.println("MSE: " + net.MSE(input2, target2));
        System.out.println("training time (ms): " + time/1000000);
    }
}


