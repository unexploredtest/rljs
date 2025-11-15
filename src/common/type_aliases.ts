import * as tf from '@tensorflow/tfjs';

// Couldn't find where TF exported this type so added it here manually
export type ActivationType = 'elu' | 'hardSigmoid' | 'linear' | 'relu' | 'relu6' | 'selu' | 'sigmoid' | 'softmax' | 'softplus' | 'softsign' | 'tanh' | 'swish' | 'mish' | 'gelu' | 'gelu_new';

export type RolloutBufferSamples = [
    observations: tf.Tensor,
    actions: tf.Tensor,
    oldValues: tf.Tensor,
    oldLogProb: tf.Tensor,
    advantages: tf.Tensor,
    returns: tf.Tensor
]

export interface BaseHyperParams {

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
    normalizeadvantage: boolean;
    batchSize: number;
    nEpochs: number;
}

export interface BaseModelParams {
    lr: number;
    activation: ActivationType;
    optim?: tf.Optimizer;
}

export interface ActorCriticParams extends BaseModelParams {
    policyNet: number[];
    valueNet: number[];
}