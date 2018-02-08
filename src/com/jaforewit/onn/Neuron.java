package com.jaforewit.onn;

import java.util.HashMap;


public class Neuron {
    private boolean canBackpropagate;
    private double bias;
    private double output;
    private double error;
    private double outputDerivative;
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight

    Neuron(double bias) {
        canBackpropagate = false;
        this.bias = bias;
        output = sigmoid(this.bias);
        outputDerivative = output * (1d - output);
        error = 0;
        outputAxons = new HashMap<>();
    }

    public double getError() { return error; }
    public double getOutput() { return output; }
    public void addOutputAxon(Neuron n, double weight) { outputAxons.put(n,weight); }
    public double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }
    public double logit(double x) { return Math.log(x / (1 - x)); }

    public void feedForward(double value, double weight) {
        canBackpropagate = true;
        output = sigmoid(logit(output) + value*weight);
        outputDerivative = output * (1 - output);

        for (Neuron n : outputAxons.keySet()) n.feedForward(output, outputAxons.get(n));

        // TODO: verify this math
    }

    public void feedForward(double value) {
        canBackpropagate = true;
        output = value;
        for (Neuron n : outputAxons.keySet()) n.feedForward(output, outputAxons.get(n));
    }

    public void setErrorFromTarget(double target) {

        // ensures feedForward() was been called
        if(!canBackpropagate) { return; }
        canBackpropagate = false;
        error = (output - target) * outputDerivative;
    }

    public void pushBackpropagation(double rate) {

        // ensures feedForward() was been called
        if(!canBackpropagate) { return; }

        canBackpropagate = false;

        // find error and adjust weights
        double sum = 0;
        double delta;
        for (Neuron n : outputAxons.keySet()) {
            n.pushBackpropagation(rate);
            sum += n.getError() * outputAxons.get(n);
            delta = -rate * output * n.getError();
            outputAxons.replace(n, outputAxons.get(n) + delta);
        }
        error = sum * outputDerivative;

        // adjust bias
        bias += -rate * error;
    }
}