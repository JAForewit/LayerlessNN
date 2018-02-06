package com.jaforewit.onn;

import java.util.HashMap;

public class Neuron {

        private int timeStep;
        private double bias;
        private double output;
        private double outputDeriv;
        private HashMap<Neuron, Double> inputs; // Key: input neuron, Value: weight


    Neuron(double bias) {
        this.bias = bias;
        timeStep = 0;
        output = 0.0;
        outputDeriv = 0.0;
        inputs = new HashMap<>();
    }

    public double getOutput() { return output; }

    public double getOutputDeriv() { return outputDeriv; }

    public void addInput(Neuron n, double weight) { inputs.put(n,weight); }

    public void manSetOutput(double output) { this.output = output; }

    public void calcOutput(int time) {
        timeStep++;
        if (inputs.isEmpty()) return;

        double sum = bias;
        for (Neuron n : inputs.keySet()) {

            if (n.timeStep < time) {
                n.calcOutput(time);
            }
            sum += (inputs.get(n) * n.getOutput());
        }

        output = sigmoid(sum);
        outputDeriv = output * (1 - output);
    }

        private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }

}