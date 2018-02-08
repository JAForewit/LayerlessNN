package com.jaforewit.onn;

import java.util.HashMap;


public class Neuron {

    private double bias;
    private double error;
    private double output;
    private double outputDerivative;
    private HashMap<Neuron, Double> inputAxons
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight

    Neuron(double bias) {
        this.bias = bias;
        error = 0d;
        output = sigmoid(this.bias);
        outputDerivative = output * (1 - output);
        inputAxons = new HashMap<>();
        outputAxons = new HashMap<>();
    }

    // Must be called if removed from an ONN
    public void close() {
        for (Neuron n : inputAxons.keySet()) removeInputAxon(n);
        for (Neuron n : outputAxons.keySet()) removeOutputAxon(n);
    }

    public void addInputAxon(Neuron n, double weight) { inputAxons.put(n,weight); }
    public void addOutputAxon(Neuron n, double weight) { outputAxons.put(n,weight); }
    public void removeInputAxon(Neuron n) {
        inputAxons.remove(n);
        n.removeOutputAxon(this);
    }
    public void removeOutputAxon(Neuron n) {
        outputAxons.remove(n);
        n.removeInputAxon(this);
    }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }
    private double logit(double x) { return Math.log(x / (1 - x)); }
}