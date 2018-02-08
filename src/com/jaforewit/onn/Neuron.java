package com.jaforewit.onn;

import java.util.HashMap;


public class Neuron {

    private int inputCounter;
    private int outputCounter;
    private double bias;
    private double error;
    private double output;
    //private double outputDerivative; // TODO: consider removing outputDerivative
    private HashMap<Neuron, Double> inputAxons; // Key: neuron, Value: weight
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight

    Neuron(double bias) {
        inputCounter = 0;
        outputCounter = 0;
        this.bias = bias;
        error = 0d;
        output = sigmoid(this.bias);
        //outputDerivative = output * (1 - output); // TODO: consider removing outputDerivative
        inputAxons = new HashMap<>();
        outputAxons = new HashMap<>();
    }

    // Must be called if removed from an ONN
    public void close() {
        for (Neuron n : inputAxons.keySet()) removeInputAxon(n);
        for (Neuron n : outputAxons.keySet()) removeOutputAxon(n);
    }

    public double getError() { return error; }

    public void addInputAxon(Neuron n, double weight) {
        inputAxons.put(n,weight);
        n.addOutputAxon(this, weight);
    }
    public void addOutputAxon(Neuron n, double weight) {
        outputAxons.put(n,weight);
        n.addInputAxon(this, weight);
    }
    public void removeInputAxon(Neuron n) {
        inputAxons.remove(n);
        n.removeOutputAxon(this);
    }
    public void removeOutputAxon(Neuron n) {
        outputAxons.remove(n);
        n.removeInputAxon(this);
    }

    public void feedForward(double value) {
        output = value;
        for (Neuron n : outputAxons.keySet()) n.feedForward(this, output);
    }

    public void backpropagate(double target, double rate) {
        error = (output - target) * output * (1 - output);
        for (Neuron n : inputAxons.keySet()) n.backpropagate(this, rate);
    }




    private void feedForward(Neuron n, double value) {
        inputCounter++;
        output = sigmoid(logit(output) + inputAxons.get(n) * value);

        if (inputCounter < inputAxons.size()) return;

        for (Neuron i : outputAxons.keySet()) i.feedForward(this, output);
        inputCounter = 0;
    }

    private void backpropagate(Neuron n, double rate) {
        outputCounter++;
        error += n.getError() * outputAxons.get(n);

        if (outputCounter < outputAxons.size()) return;

        error *= output * (1 - output);

        // update weights
        double newWeight;
        for (Neuron i : outputAxons.keySet()) {
            newWeight = outputAxons.get(i) - rate * output * i.getError();
            removeOutputAxon(i);
            addInputAxon(i, newWeight);
        }

        for (Neuron i : inputAxons.keySet()) i.backpropagate(this, rate);
        outputCounter = 0;
    }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }

    private double logit(double x) { return Math.log(x / (1 - x)); }
}