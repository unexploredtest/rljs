import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces';

import { OnPolicyAlgorithm } from '../common/on_policy_algorithm.js';


export class PPO extends OnPolicyAlgorithm {
    private clipRange: number;
    private batchSize: number;
    private nEpochs: number;
    constructor(env: gym.Env | gym.Wrapper,
        lr: number = 3e-4, nSteps: number = 2048, gamma: number = 0.99, gaeLmbda: number = 0.95,
        entCoef: number = 0.2, vfCoef: number = 0.5, maxGradNorm: number = 0.5, clipRange: number = 0.2,
        batchSize: number = 64, nEpochs: number = 10) {
        super(env, lr, nSteps, gamma, gaeLmbda, entCoef, vfCoef, maxGradNorm);

        this.clipRange = clipRange;
        this.batchSize = batchSize;
        this.nEpochs = nEpochs;
    }

    async train(): Promise<void> {
        let entropyLosses: tf.Tensor[] = [];
        let pgLosses: tf.Tensor[] = [];
        let valueLosses: tf.Tensor[] = [];
        let clipFractions: tf.Tensor[] = [];

        let batches = this.buffer.get(this.buffer.size);

        for(let i = 0; i < this.nEpochs; i++) {
            let approxKlDivs: tf.Tensor[] = [];

            // for(let j = 0; j < batches.length; j++) {
            console.log(i + " : " + batches.length + " : ");
            let [observations, actions, oldValues, oldLogProb, advantages, returns] = batches[0];;

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

            const result = this.policy.optim.computeGradients(optFunc);
            // https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
            let normedGrad: tf.NamedTensorMap = {};
            for (const key in result.grads) {
                let tensorGrad = result.grads[key];
                let totalNorm = tf.norm(tensorGrad);
                let clipCoef = tf.scalar(this.maxGradNorm).div(totalNorm.add(tf.scalar(1e-6)))
                let clipCoefClamped = tf.clipByValue(clipCoef, 0, 1);
                normedGrad[key] = tensorGrad.mul(clipCoefClamped);
            }
            
            this.policy.optim.applyGradients(normedGrad);
            result.grads

            // batches.forEach((batch) => {
            //     tf.dispose(batch);
            // })
            // }
        }

        tf.dispose(batches);
    }
}