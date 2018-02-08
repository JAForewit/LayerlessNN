package OrganicNN;

/**
 * Created by JAForewit on 02.08.2017
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import TrainSet.TrainSet;


public class ONN {
    private static final Logger LOGGER = Logger.getLogger( ONN.class.getName() );
    private final double MIN_BIAS = -0.7;
    private final double MAX_BIAS = 0.7;
    private final double MIN_WEIGHT = -1.0;
    private final double MAX_WEIGHT = 1.0;
    private int inputCount;     // number of input neurons
    private int outputCount;    // number of output neurons
    private int neuronCount;    // total number of neurons
    private Neuron neurons[];
    private double[] latestOutputs;


    public ONN(String filename) throws IOException {
        // load structure file
        try(BufferedReader reader = new BufferedReader(new FileReader(filename))) {

            // read network critical definitions
            int[] nextLine = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();
            inputCount = nextLine[0];
            outputCount = nextLine[1];
            neuronCount = inputCount + outputCount + nextLine[2];
            neurons = new Neuron[neuronCount];
            latestOutputs = new double[outputCount];

            // initialize neurons with a random bias
            double randBias;
            for (int i = 0; i < neuronCount; i++) {
                randBias = (Math.random() * (MAX_BIAS - MIN_BIAS)) + MIN_BIAS;
                neurons[i] = new Neuron(randBias);
            }

            // create axons and assign random weights
            double randWeight;
            for (int i = 0; i < neuronCount - outputCount; i++) {
                nextLine = Arrays.stream(reader.readLine().split("\\s")).mapToInt(Integer::parseInt).toArray();
                for (int index : nextLine) {
                    randWeight = (Math.random() * (MAX_WEIGHT - MIN_WEIGHT)) + MIN_WEIGHT;
                    neurons[i].addOutputAxon(neurons[index], randWeight);
                }
            }

            // verifying the file has ended
            if (reader.readLine() != null) throw new IOException();
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "There was a problem reading the structure file.");
            throw e;
        }
    }

    public double[] calculateOutputs(double[] inputs) throws InvalidParameterException {
        if (inputs.length != inputCount) {
            LOGGER.log(Level.SEVERE, "Passed an invalid input size to calculateOutputs()."
                    + " Expected inputs[" + inputCount + "].");
            throw new InvalidParameterException();
        }
        feedForward(inputs);
        for (int i=0; i<outputCount; i++) latestOutputs[i] = neurons[neuronCount-i-1].getOutput();
        return latestOutputs;
    }

    public void train(double[] inputs, double[] targets, double rate, int iterations) throws InvalidParameterException {
        if (inputs.length != inputCount || targets.length != outputCount) {
            LOGGER.log(Level.SEVERE, "Passed an invalid input and target size to train()."
                    + " Expected inputs[" + inputCount + "], targets[" + outputCount + "].");
            throw new InvalidParameterException();
        }
        for (int i=0; i < iterations; i++) backpropagate(inputs,targets,rate);
    }

    public void train(TrainSet set, int loops, int batch_size) throws InvalidParameterException{
        if (set.getINPUT_SIZE() != inputCount || set.getOUTPUT_SIZE() != outputCount) {
            LOGGER.log(Level.SEVERE, "Passed a TrainSet with an invalid input and target size to train()."
                    + " Expected inputs[" + inputCount + "], targets[" + outputCount + "].");
            throw new InvalidParameterException();
        }
        for (int i = 0; i < loops; i++) {
            TrainSet batch = set.extractBatch(batch_size);
            for (int j=0; j < batch_size; j++) {
                this.backpropagate(batch.getInput(j), batch.getOutput(j), 0.3);
            }
        }
    }

    public double MSE (double[] inputs, double[] targets) throws InvalidParameterException {
        if (inputs.length != inputCount || targets.length != outputCount) {
            LOGGER.log(Level.SEVERE, "MSE(): passed an invalid input and target size."
                    + " Expected: inputs[" + inputCount + "] targets[" + outputCount + "].");
            throw new InvalidParameterException();
        }
        double sum = 0;
        for (int i=0; i<outputCount; i++)
            sum += (targets[i] - calculateOutputs(inputs)[i]) * (targets[i] - calculateOutputs(inputs)[i]);
        return sum / (2d * outputCount);
    }


    private void feedForward(double[] inputs) {
        for (int i=0; i<inputCount; i++) neurons[i].feedForward(inputs[i]);
    }

    private void backpropagate(double[] inputs, double[] targets, double rate) {
        feedForward(inputs);
        for (int i=0; i<outputCount; i++) neurons[neuronCount-i-1].backpropagate(targets[i], rate);
    }
}