import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';

import { OnPolicyAlgorithm } from '../common/on_policy_algorithm.js';
import { clipGradNorm } from '../common/utils.js';
import { PPOHyperParameters, ActorCriticParams } from '../common/type_aliases.js';

const DEFAULT_HYPER_PARAMETERS: PPOHyperParameters = {
    nSteps: 2048,
    gamma: 0.99,
    gaeLmbda: 0.95,
    entCoef: 0.2,
    vfCoef: 0.5,
    maxGradNorm: 0.5,
    clipRange: 0.2,
    normalizeadvantage: true,
    batchSize: 64,
    nEpochs: 10
}

const DEFAULT_POLICY_PARAMETERS: ActorCriticParams = {
    lr: 3e-4,
    activation: "relu",
    optim: undefined,
    policyNet: [64, 64],
    valueNet: [64, 64],
    logStdInit: 0.0
}


export class PPO extends OnPolicyAlgorithm {
    private clipRange: number;
    private batchSize: number;
    private nEpochs: number;
    private normalizeadvantage: boolean;
    private target_kl: number | undefined;

    constructor(env: gym.Env | gym.Wrapper, hyperParameters: Partial<PPOHyperParameters> = {}, policyParameters: Partial<ActorCriticParams> = {}) {
        const fHyperParameters = Object.assign({}, DEFAULT_HYPER_PARAMETERS, hyperParameters) as PPOHyperParameters;
        const fPolicyParameters = Object.assign({}, DEFAULT_POLICY_PARAMETERS, policyParameters) as ActorCriticParams;
        super(env, fHyperParameters, fPolicyParameters);
        this.clipRange = fHyperParameters.clipRange;
        this.normalizeadvantage = fHyperParameters.normalizeadvantage;
        this.batchSize = fHyperParameters.batchSize;
        this.nEpochs = fHyperParameters.nEpochs;
        this.target_kl = fHyperParameters.target_kl;

        if(this.normalizeadvantage && this.batchSize === 1) {
            throw new Error("Batch size must be greater than 1");
        }
    }

    async train(): Promise<void> {
        let entropyLosses: number[] = [];
        let pgLosses: number[] = [];
        let valueLosses: number[] = [];
        let clipFractions: number[] = [];
        let approxKlDivs: number[] = [];
        let lossValue: number = 0;

        for(let i = 0; i < this.nEpochs; i++) {
            approxKlDivs = [];
            let continueTraining = true;

            for (let batch of this.buffer.get(this.batchSize)) {
                let [observations, actions, oldValues, oldLogProb, advantages, returns] = batch;

                if(this.normalizeadvantage && advantages.size > 1) {
                    advantages = tf.tidy(() => {
                        const {mean, variance} = tf.moments(advantages); 
                        const std = variance.sqrt();
                        return (advantages.sub(mean)).div(std.add(tf.scalar(1e-8)));
                    })
                }

                let approxKlDiv: number;

                const optFunc = () => {
                    let [values, logProb, entropy] = this.policy.evaluateActions(observations, actions);
                    values = values.flatten();
                    let ratio = tf.exp(logProb.sub(oldLogProb));
                    let policyLoss1 = advantages.mul(ratio);
                    let policyLoss2 = advantages.mul(tf.clipByValue(ratio, 1 - this.clipRange, 1 + this.clipRange));
                    let policyLoss = tf.neg(tf.minimum(policyLoss1, policyLoss2)).mean();
                    pgLosses.push(policyLoss.asScalar().dataSync()[0])

                    const clipFraction = tf.mean(tf.scalar(this.clipRange).less((tf.abs(ratio.sub(tf.scalar(1)))).toFloat()));
                    clipFractions.push(clipFraction.asScalar().dataSync()[0]);

                    let valueLoss = (returns.sub(values)).square().mean();
                    valueLosses.push(valueLoss.asScalar().dataSync()[0])

                    let entropyLoss = tf.neg(entropy.mean());
                    entropyLosses.push(entropyLoss.asScalar().dataSync()[0])

                    let loss = policyLoss.add(entropyLoss.mul(this.entCoef)).add(valueLoss.mul(this.vfCoef));
                    lossValue = loss.asScalar().dataSync()[0];

                    const logRatio = logProb.sub(oldLogProb);
                    const approxKlDivScalar = tf.mean((tf.exp(logRatio).sub(tf.scalar(1)).sub(logRatio)));
                    approxKlDiv = approxKlDivScalar.asScalar().dataSync()[0];
                    approxKlDivs.push(approxKlDiv);

                    return loss.asScalar();
                }

                tf.tidy(() => {                 
                    const result = this.policy.optim.computeGradients(optFunc);
                    let normedGrad: tf.NamedTensorMap = clipGradNorm(result.grads, this.maxGradNorm);                    
                    this.policy.optim.applyGradients(normedGrad);

                    if(this.target_kl !== undefined && approxKlDiv > 1.5 * this.target_kl) {
                        continueTraining = false;
                    }
                })

                tf.dispose(batch);

                if(!continueTraining) {
                    break;
                }
            }

            if(!continueTraining) {
                break;
            }
        }

        this.logger.record("Entropy Loss", tf.util.sum(entropyLosses) / entropyLosses.length);
        this.logger.record("Policy Gradient Loss", tf.util.sum(pgLosses) / pgLosses.length);
        this.logger.record("Value Loss", tf.util.sum(valueLosses) / valueLosses.length);
        this.logger.record("Approximate KL", tf.util.sum(approxKlDivs) / approxKlDivs.length);
        this.logger.record("Clip Fraction", tf.util.sum(clipFractions) / clipFractions.length);
        this.logger.record("Loss", lossValue);

        const logStd = this.policy.getLogStd();
        if(logStd) {
            this.logger.record("Log STD", tf.exp(logStd).mean().dataSync()[0]);
        }
    }
}