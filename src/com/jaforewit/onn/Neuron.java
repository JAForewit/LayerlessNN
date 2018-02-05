package com.jaforewit.onn;

import java.util.HashMap;

public class Neuron {

    private double bias;
    private double output;
    private double outputDeriv;
    private HashMap<Neuron, Double> inputs; // Key: input neuron, Value: weight


    Neuron(double bias) {
        this.bias = bias;
        output = 0.0;
        outputDeriv = 0.0;
        inputs = new HashMap<>();
    }


    public double getOutput() { calcOutput(); return output; }

    public double getOutputDeriv() { calcOutput(); return outputDeriv; }

    public void addInput(Neuron n, double weight) { inputs.put(n,weight); }

    public void setOutput(double output) { this.output = output; }



    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }

    private void calcOutput() {
        if (inputs.isEmpty()) return;

        double sum = bias;
        for (Neuron n : inputs.keySet()) {
            sum += (inputs.get(n) * n.getOutput());
        }

        output = sigmoid(sum);
        outputDeriv = output * (1 - output);
    }
}