package com.jaforewit.onn;

import java.util.HashMap;

public class Neuron {

    private double bias;
    private double output;
    private double outputDerivative;
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight

    Neuron(double bias) {
        this.bias = bias;
        output = sigmoid(this.bias);
        outputDerivative = output * (1 - output);
        outputAxons = new HashMap<>();
    }

    public double getOutput() { return output; }

    public double getOutputDerivative() { return outputDerivative; }

    public void addOutput(Neuron n, double weight) { outputAxons.put(n,weight); }

    public void feedForward(double value, double weight) {
        output = sigmoid(Math.log(output/(1-output)) + value*weight);
        outputDerivative = output * (1 - output);

        for (Neuron n : outputAxons.keySet()) n.feedForward(output, outputAxons.get(n));

        // TODO: add math to README.md
    }

    public void feedForward(double value) {
        output = value;
        for (Neuron n : outputAxons.keySet()) n.feedForward(output, outputAxons.get(n));
    }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }
}