package com.jaforewit.onn;

import java.util.HashMap;


public class Neuron {

    private double output;
    private double outputDerivative;
    private double bias;
    private double error;
    private HashMap<Neuron, Double> inputAxons
    private HashMap<Neuron, Double> outputAxons; // Key: neuron, Value: weight

    Neuron(double bias) {

    }

    private double sigmoid(double x) { return 1d / (1 + Math.exp(-x)); }
    private double logit(double x) { return Math.log(x / (1 - x)); }

}