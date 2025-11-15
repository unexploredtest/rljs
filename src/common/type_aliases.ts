import * as tf from '@tensorflow/tfjs';

export type RolloutBufferSamples = [
    observations: tf.Tensor,
    actions: tf.Tensor,
    oldValues: tf.Tensor,
    oldLogProb: tf.Tensor,
    advantages: tf.Tensor,
    returns: tf.Tensor
]

export interface BaseHyperParams {
    lr: number;
}

export interface OnPolicyHyperParams extends BaseHyperParams {
    nSteps: number;
    gamma: number;
    gaeLmbda: number;
    entCoef: number;
    vfCoef: number;
    maxGradNorm: number;
}

export interface PPOHyperParameters extends OnPolicyHyperParams {
    clipRange: number;
    batchSize: number;
    nEpochs: number;
}