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
    init(inputCount: Int, hiddenCount: Int, outputCount: Int, learnRate: Double, momentum: Double) {
        self.learnRate = learnRate
        self.momentum = momentum
        self.inputCount = inputCount
        self.hiddenCount = hiddenCount
        self.outputCount = outputCount

        neuronCount = inputCount + hiddenCount + outputCount
        weightCount = (inputCount * hiddenCount) + (hiddenCount * outputCount)
        
        fire        = Array<Double>(repeating: 0.0, count: neuronCount)
        matrix      = Array<Double>(repeating: 0.5 - drand48(), count: weightCount)
        matrixDelta = Array<Double>(repeating: 0.0, count: weightCount)
        thresholds  = Array<Double>(repeating: 0.5 - drand48(), count: neuronCount)
        errorDelta  = Array<Double>(repeating: 0.0, count: neuronCount)
        error       = Array<Double>(repeating: 0.0, count: neuronCount)
        accThresholdDelta = Array<Double>(repeating: 0.0, count: neuronCount)
        accMatrixDelta = Array<Double>(repeating: 0.0, count: weightCount)
        thresholdDelta = Array<Double>(repeating: 0.0, count: neuronCount)
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
    public mutating func computeOutputs(input: [Double]) -> [Double] {
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
    public mutating func calcError(ideal: [Double]) {
    }


    /// Modify the weight matrix and thresholds based on the last call to calcError.
    public mutating func learn() {
    }
}


