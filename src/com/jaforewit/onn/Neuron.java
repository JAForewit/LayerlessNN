package com.jaforewit.onn;

import java.util.HashMap;
import java.util.List;

public class Neuron {

    private double bias;
    private double output;
    private double outputDeriv;
    private HashMap<Neuron, Double> outputs; // Key: input neuron, Value: weight

    Neuron(double bias) {
        this.bias = bias;
        output = sigmoid(this.bias);
        outputDeriv = output * (1 - output);
        outputs = new HashMap<>();
    }

    public double getOutput() { return output; }

    public double getOutputDeriv() { return outputDeriv; }

    public void addOutput(Neuron n, double weight) { outputs.put(n,weight); }

    public void feedForward(double value, double weight) {
        output = sigmoid(Math.log(output/(1-output)) + value*weight);
        outputDeriv = output * (1 - output);

        for (Neuron n : outputs.keySet()) n.feedForward(output, outputs.get(n));

        // TODO: add math to README.md
    }

    public void feedForward(double value) {
        output = value;
        for (Neuron n : outputs.keySet()) n.feedForward(output, outputs.get(n));
    }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }
}