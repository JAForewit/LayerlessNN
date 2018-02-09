
import java.util.Arrays;
import OrganicNN.*;
import TrainSet.*;

public class Main {
    public static void main(String[] args) {

        try {
            OrganicNN net = new OrganicNN("test.structure");
            TrainSet trainingSet = new TrainSet(net.getInputCount(),net.getOutputCount());




        } catch (Exception e) { e.printStackTrace(); }

    }
}


