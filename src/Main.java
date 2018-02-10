
import OrganicNN.*;
import TrainSet.*;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        try {
            OrganicNN net = new OrganicNN("test.structure");
            TrainSet trainSet = new TrainSet(5,1);


            double[] test4 = {2d,-2d,2d,-2d,2d}, out4 = {0};
            double[] test5 = {-2d,2d,-2d,2d,-2d}, out5 = {1};


            for (int i=0; i< 10000; i++) {
                net.train(test4,out4,0.3,1);
                net.train(test5,out5,0.3,1);
            }

            System.out.println(Arrays.toString(net.getOutputs(test4)));
            System.out.println(Arrays.toString(net.getOutputs(test5)));
        } catch (Exception e) { e.printStackTrace(); }

    }
}


