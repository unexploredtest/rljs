import * as tfCore from '@tensorflow/tfjs';

// Couldn't find where TF exported this type so added it here manually
export type ActivationType =
  | 'elu'
  | 'hardSigmoid'
  | 'linear'
  | 'relu'
  | 'relu6'
  | 'selu'
  | 'sigmoid'
  | 'softmax'
  | 'softplus'
  | 'softsign'
  | 'tanh'
  | 'swish'
  | 'mish'
  | 'gelu'
  | 'gelu_new';

export type RolloutBufferSamples = [
  observations: tfCore.Tensor,
  actions: tfCore.Tensor,
  oldValues: tfCore.Tensor,
  oldLogProb: tfCore.Tensor,
  advantages: tfCore.Tensor,
  returns: tfCore.Tensor,
];

export type RolloutBufferResult = {
  observations: tfCore.Tensor;
  actions: tfCore.Tensor;
  values: tfCore.Tensor;
  logProb: tfCore.Tensor;
  advantages: tfCore.Tensor;
};

export type ReplayBufferSamples = [
  observations: tfCore.Tensor,
  nextObservations: tfCore.Tensor,
  actions: tfCore.Tensor,
  rewards: tfCore.Tensor,
  dones: tfCore.Tensor,
  timeouts: tfCore.Tensor,
];

export interface BaseHyperParams {}

export interface OnPolicyHyperParams extends BaseHyperParams {
  nSteps: number;
  gamma: number;
  gaeLmbda: number;
  entCoef: number;
  vfCoef: number;
  maxGradNorm: number;
}

export interface OffPolicyHyperParams extends BaseHyperParams {
  bufferSize: number;
  learningStarts: number;
  batchSize: number;
  tau: number;
  gamma: number;
  trainFreq: number;
  gradientSteps: number;
  nSteps: number;
}

export interface PPOHyperParameters extends OnPolicyHyperParams {
  clipRange: number;
  normalizeadvantage: boolean;
  batchSize: number;
  nEpochs: number;
  target_kl?: number;
}

export interface A2CHyperParameters extends OnPolicyHyperParams {
  normalizeadvantage: boolean;
}

export interface DQNHyperParams extends OffPolicyHyperParams {
  maxGradNorm: number;
  initialEps: number;
  finalEps: number;
  explorationFraction: number;
  targetUpdateInterval: number;
}

export interface BaseModelParams {
  lr: number;
  activation: ActivationType;
  optim?: tfCore.Optimizer;
}

export interface ActorCriticParams extends BaseModelParams {
  policyNet: number[];
  valueNet: number[];
  logStdInit: number;
}

export interface QNetworkParams extends BaseModelParams {
  net: number[];
}
