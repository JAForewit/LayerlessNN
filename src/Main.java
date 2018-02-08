
import java.util.Arrays;
import OrganicDNN.*;
import TrainSet.*;

public class Main {
    public static void main(String[] args) {

        try {
            OrganicDNN net = new OrganicDNN("test.structure");
            TrainSet trainingSet = new TrainSet(net.getInputCount(),net.getOutputCount());




        } catch (Exception e) { e.printStackTrace(); }

    }
}


