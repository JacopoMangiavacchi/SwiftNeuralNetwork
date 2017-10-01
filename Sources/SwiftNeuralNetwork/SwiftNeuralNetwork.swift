/**
 * Network
 * Copyright 2017 by Jacopo Mangiavacchi (jmangia@me.com)
 * Ported from original Java code Copyright 2005 by Jeff Heaton(jeff@jeffheaton.com)
 *
 * This software is copyrighted. You may use it in programs
 * of your own, without restriction, but you may not
 * publish the source code without the author's permission.
 */

import Foundation


/// Neural Network struct
public struct Network {
    
    /// The global error for the training.
    private var globalError: Double = 0.0
    
    /// The number of input neurons.
    private let inputCount: Int
    
    /// The number of hidden neurons.
    private let hiddenCount: Int
    
    /// The number of output neurons
    private let outputCount: Int
    
    /// The total number of neurons in the network.
    private var neuronCount: Int
    
    /// The number of weights in the network.
    private var weightCount: Int
    
    /// The learning rate.
    private let learnRate: Double
    
    /// The outputs from the various levels.
    private var fire: [Double]
    
    /// The weight matrix this, along with the thresholds can be thought of as the "memory" of the neural network.
    private var matrix: [Double]
    
    /// The errors from the last calculation.
    private var error: [Double]
    
    /// Accumulates matrix delta's for training.
    private var accMatrixDelta: [Double]
    
    /// The thresholds, this value, along with the weight matrix can be thought of as the memory of the neural network.
    private var thresholds: [Double]
    
    /// The changes that should be applied to the weight matrix.
    private var matrixDelta: [Double]
    
    /// The accumulation of the threshold deltas.
    private var accThresholdDelta: [Double]
    
    /// The threshold deltas.
    private var thresholdDelta: [Double]
    
    /// The momentum for training.
    private let momentum: Double
    
    /// The changes in the errors.
    private var errorDelta: [Double]
    
    
    /**
     Init the neural network.
     
     - Parameter inputCount: The number of input neurons.
     - Parameter hiddenCount: The number of hidden neurons
     - Parameter outputCount: The number of output neurons
     - Parameter learnRate: The learning rate to be used when training.
     - Parameter momentum: The momentum to be used when training.
     */
    public init(inputCount: Int, hiddenCount: Int, outputCount: Int, learnRate: Double, momentum: Double) {
        self.learnRate = learnRate
        self.momentum = momentum
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount
        
        neuronCount = inputCount + hiddenCount + outputCount
        weightCount = (inputCount * hiddenCount) + (hiddenCount * outputCount)
        
        fire        = Array<Double>(repeating: 0.0, count: neuronCount)
        matrix      = Array<Double>(repeating: 0.0, count: weightCount)
        matrixDelta = Array<Double>(repeating: 0.0, count: weightCount)
        thresholds  = Array<Double>(repeating: 0.0, count: neuronCount)
        errorDelta  = Array<Double>(repeating: 0.0, count: neuronCount)
        error       = Array<Double>(repeating: 0.0, count: neuronCount)
        accThresholdDelta = Array<Double>(repeating: 0.0, count: neuronCount)
        accMatrixDelta = Array<Double>(repeating: 0.0, count: weightCount)
        thresholdDelta = Array<Double>(repeating: 0.0, count: neuronCount)

        for i in 0..<weightCount {
            matrix[i] = 0.5 - drand48()
        }

        for i in 0..<neuronCount {
            thresholds[i] = 0.5 - drand48()
        }
}
    
    
    /**
     Train the Neural Network with a set of data
     
     - Parameter input: The input provide to the neural network.
     - Parameter ideal: What the output neurons should have yielded.
     */
    public mutating func train(input: [Double], ideal: [Double]) {
        _ = computeOutputs(input: input)
        calcError(ideal: ideal)
        learn()
    }
    
    
    /**
     Predict using the Neural Network
     
     - Parameter input: The input provide to the neural network.
     - Returns: The results from the output neurons.
     */
    public mutating func predict(input: [Double]) -> [Double] {
        return computeOutputs(input: input)
    }
    
    
    
    /**
      Convert to an Data buffer to be used for "memory" persistence of the neurons
      (the weight and threshold values) be expressed as a linear array.
     
      - Returns: The memory of the neuron.
     */
    public func save() -> Data {
//TODO: conversion
//        double result[] = new double[matrix.length+thresholds.length];
//        for (int i=0;i<matrix.length;i++)
//            result[i] = matrix[i];
//        for (int i=0;i<thresholds.length;i++)
//            result[matrix.length+i] = thresholds[i];
//        return result;
        return Data()
    }

    
    
    /**
     Convert from a Data buffer used for "memory" persistence of the neurons
     (the weight and threshold values) be expressed as a linear array.
     
     - Parameter data: Data buffer for "memory" persistence of the neurons
     */
    public mutating func load(data: Data) {
//TODO: conversion
//        for (int i=0;i<matrix.length;i++)
//            matrix[i] = array[i];
//        for (int i=0;i<thresholds.length;i++)
//            thresholds[i] = array[matrix.length+i];
    }
    
    
    
    /**
     Returns the root mean square error for a complete training set.
     
     - Parameter len: The length of a complete training set.
     - Returns: The current error for the neural network.
     */
    public mutating func getError(len: Int) -> Double {
        let err = sqrt(globalError / Double(len * outputCount))
        globalError = 0.0  // clear the accumulator
        return err
    }
    
    
    /**
     The threshold method. You may wish to override this class to provide other
     threshold methods.
     
     - Parameter sum: The activation from the neuron..
     - Returns: The activation applied to the threshold method.
     */
    private func threshold(sum: Double) -> Double {
        return 1.0 / (1 + exp(-1.0 * sum))
    }
    
    
    /**
     Compute the output for a given input to the neural network.
     
     - Parameter input: The input provide to the neural network.
     - Returns: The results from the output neurons.
     */
    private mutating func computeOutputs(input: [Double]) -> [Double] {
        for i in 0..<inputCount {
            fire[i] = input[i]
        }
        
        // first layer
        var inx = 0
        
        for i in 0..<hiddenCount {
            let i2 = i + inputCount
            var sum = thresholds[i2]
            
            for j in 0..<inputCount {
                sum += fire[j] * matrix[inx]
                inx += 1
            }
            fire[i2] = threshold(sum: sum)
        }
        
        // hidden layer
        var result = Array<Double>(repeating: 0.0, count: outputCount)
        
        for i in (inputCount + hiddenCount)..<neuronCount {
            var sum = thresholds[i]
            
            for j in 0..<hiddenCount {
                let j2 = j + inputCount
                sum += fire[j2] * matrix[inx]
                inx += 1
            }
            fire[i] = threshold(sum: sum)
            
            result[i - inputCount - hiddenCount] = fire[i]
        }
        
        return result;
    }
    
    
    /**
     Calculate the error for the recognition just done.
     
     - Parameter ideal: What the output neurons should have yielded.
     */
    private mutating func calcError(ideal: [Double]) {
        // clear hidden layer errors
        for i in inputCount..<neuronCount {
            error[i] = 0.0
        }
        
        // layer errors and deltas for output layer
        for i in (inputCount + hiddenCount)..<neuronCount {
            error[i] = ideal[i - inputCount - hiddenCount] - fire[i]
            globalError += error[i] * error[i]
            errorDelta[i] = error[i] * fire[i] * (1 - fire[i])
        }
        
        // hidden layer errors
        var winx = inputCount * hiddenCount
        
        for i in (inputCount + hiddenCount)..<neuronCount {
            for j in 0..<hiddenCount {
                let j2 = j + inputCount
                accMatrixDelta[winx] += errorDelta[i] * fire[j2]
                error[j2] += matrix[winx] * errorDelta[i]
                winx += 1
            }
            accThresholdDelta[i] += errorDelta[i]
        }
        
        // hidden layer deltas
        for i in 0..<hiddenCount {
            let i2 = i + inputCount
            errorDelta[i2] = error[i2] * fire[i2] * (1.0 - fire[i2])
        }
        
        // input layer errors
        winx = 0  // offset into weight array
        for i in 0..<hiddenCount {
            let i2 = i + inputCount
            for j in 0..<inputCount {
                accMatrixDelta[winx] += errorDelta[i2] * fire[j]
                error[j] += matrix[winx] * errorDelta[i2]
                winx += 1
            }
            accThresholdDelta[i2] += errorDelta[i2]
        }
    }
    
    
    /// Modify the weight matrix and thresholds based on the last call to calcError.
    private mutating func learn() {
        // process the matrix
        for i in 0..<matrix.count {
            matrixDelta[i] = (learnRate * accMatrixDelta[i]) + (momentum * matrixDelta[i])
            matrix[i] += matrixDelta[i]
            accMatrixDelta[i] = 0
        }
        
        // process the thresholds
        for i in inputCount..<neuronCount {
            thresholdDelta[i] = learnRate * accThresholdDelta[i] + (momentum * thresholdDelta[i])
            thresholds[i] += thresholdDelta[i]
            accThresholdDelta[i] = 0
        }
    }
}

