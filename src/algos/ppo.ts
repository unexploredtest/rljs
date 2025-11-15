import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces';

import { OnPolicyAlgorithm } from '../common/on_policy_algorithm.js';
import { clipGradNorm } from '../common/utils.js';
import { PPOHyperParameters } from '../common/type_aliases.js';

const DEFAULT_HYPER_PARAMETERS: PPOHyperParameters = {
    lr: 3e-4,
    nSteps: 2048,
    gamma: 0.99,
    gaeLmbda: 0.95,
    entCoef: 0.2,
    vfCoef: 0.5,
    maxGradNorm: 0.5,
    clipRange: 0.2,
    batchSize: 64,
    nEpochs: 10
}

// constructor(env: gym.Env | gym.Wrapper,
//     lr: number = 3e-4, nSteps: number = 2048, gamma: number = 0.99, gaeLmbda: number = 0.95,
//     entCoef: number = 0.2, vfCoef: number = 0.5, maxGradNorm: number = 0.5, policyNet: number[] = [64, 64], valueNet: number[] = [64, 64], clipRange: number = 0.2,
//     batchSize: number = 64, nEpochs: number = 10) {


export class PPO extends OnPolicyAlgorithm {
    private clipRange: number;
    private batchSize: number;
    private nEpochs: number;
    constructor(env: gym.Env | gym.Wrapper, hyperParameters: Partial<PPOHyperParameters> = {}, policyNet: number[] = [64, 64], valueNet: number[] = [64, 64]) {
        const fHyperParameters = Object.assign({}, DEFAULT_HYPER_PARAMETERS, hyperParameters) as PPOHyperParameters;
        super(env, fHyperParameters, policyNet, valueNet);
        this.clipRange = fHyperParameters.clipRange;
        this.batchSize = fHyperParameters.batchSize;
        this.nEpochs = fHyperParameters.nEpochs;
    }

    async train(): Promise<void> {
        let entropyLosses: tf.Tensor[] = [];
        let pgLosses: tf.Tensor[] = [];
        let valueLosses: tf.Tensor[] = [];
        let clipFractions: tf.Tensor[] = [];

        for(let i = 0; i < this.nEpochs; i++) {
            let approxKlDivs: tf.Tensor[] = [];

            for (let batch of this.buffer.get(this.batchSize)) {
                let [observations, actions, oldValues, oldLogProb, advantages, returns] = batch;

                const optFunc = () => {
                    let [values, logProb, entropy] = this.policy.evaluateActions(observations, actions);
                    values = values.flatten();
                    let ratio = tf.exp(logProb.sub(oldLogProb));
                    let policyLoss1 = advantages.mul(ratio);
                    let policyLoss2 = advantages.mul(tf.clipByValue(ratio, 1 - this.clipRange, 1 + this.clipRange));
                    let policyLoss = tf.neg(tf.minimum(policyLoss1, policyLoss2)).mean();

                    let valueLoss = (returns.sub(values)).square().mean().mul(this.vfCoef);

                    let entropyLoss = tf.neg(entropy.mean()).mul(tf.scalar(this.entCoef));

                    let loss = policyLoss.add(entropyLoss).add(valueLoss);
                    return loss.asScalar();
                }

                tf.tidy(() => {                 
                    const result = this.policy.optim.computeGradients(optFunc);
                    let normedGrad: tf.NamedTensorMap = clipGradNorm(result.grads, this.maxGradNorm);                    
                    this.policy.optim.applyGradients(normedGrad);
                })
                tf.dispose(batch);
            }
        }
    }
}