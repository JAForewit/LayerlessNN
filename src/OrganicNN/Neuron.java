package OrganicNN;

import java.util.HashMap;

/**
 * Created by JAForewit on 02.08.2017
 */

public class Neuron {
    private int inputCounter;
    private int outputCounter;
    private double bias;
    private double error;
    private double output;
    private HashMap<Neuron, Double> inputAxons; // Key: neuron, Value: weight
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight


    public Neuron(double bias) {
        inputCounter = 0;
        outputCounter = 0;
        this.bias = bias;
        error = 0d;
        output = sigmoid(this.bias);
        inputAxons = new HashMap<>();
        outputAxons = new HashMap<>();
    }

    // Must be called if removed from an ONN
    public void close() {
        for (Neuron n : inputAxons.keySet()) removeInputAxon(n);
        for (Neuron n : outputAxons.keySet()) removeOutputAxon(n);
    }

    public double getOutput() { return output; }

    public void addInputAxon(Neuron n, double weight) {
        if (inputAxons.containsKey(n)) return;
        inputAxons.put(n,weight);
        n.addOutputAxon(this, weight);
    }

    public void addOutputAxon(Neuron n, double weight) {
        if (outputAxons.containsKey(n)) return;
        outputAxons.put(n,weight);
        n.addInputAxon(this, weight);
    }

    public void removeInputAxon(Neuron n) {
        if (inputAxons.containsKey(n)) {
            inputAxons.remove(n);
            n.removeOutputAxon(this);
        }
    }

    public void removeOutputAxon(Neuron n) {
        if (outputAxons.containsKey(n)) {
            outputAxons.remove(n);
            n.removeInputAxon(this);
        }
    }

    public void updateInputAxon(Neuron n, double weight) {
        if (inputAxons.get(n) == weight) return;
        inputAxons.replace(n, weight);
        n.updateOutputAxon(this, weight);
    }

    public void updateOutputAxon(Neuron n, double weight) {
        if (outputAxons.get(n) == weight) return;
        outputAxons.replace(n, weight);
        n.updateInputAxon(this, weight);
    }

    public void feedForward(double value) {
        output = value;

        // continue feeding forward
        for (Neuron n : outputAxons.keySet()) n.feedForward(this);

        // reset output
        output = sigmoid(bias);
    }

    public void backpropagate(double target, double rate) {
        error = (output - target) * output * (1 - output);

        // continue backpropagation
        for (Neuron n : inputAxons.keySet()) n.backpropagate(this, rate);

        // update bias and reset output
        bias += -rate * error;
        output = sigmoid(bias);
    }


    private void feedForward(Neuron n) {
        inputCounter++;
        output = sigmoid(logit(output) + inputAxons.get(n) * n.getOutput());

        // ensure all input axons have been fed forward
        if (inputCounter < inputAxons.size()) return;

        // continue feeding forward
        for (Neuron i : outputAxons.keySet()) i.feedForward(this);

        // reset output
        output = sigmoid(bias);
        inputCounter = 0;
    }

    private void backpropagate(Neuron n, double rate) {
        outputCounter++;
        error += n.getError() * outputAxons.get(n);

        // ensure all output axons have backpropagated
        if (outputCounter < outputAxons.size()) return;

        error *= output * (1 - output);

        // update weights
        double newWeight;
        for (Neuron i : outputAxons.keySet()) {
            newWeight = outputAxons.get(i) - rate * output * i.getError();
            updateOutputAxon(i, newWeight);
        }

        // continue backpropagation
        for (Neuron i : inputAxons.keySet()) i.backpropagate(this, rate);

        // update bias and reset output
        bias += -rate * error;
        output = sigmoid(bias);
        outputCounter = 0;
    }

    private double getError() { return error; }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }

    private double logit(double x) { return Math.log(x / (1 - x)); }
}