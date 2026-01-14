import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';
import * as gym from '@unexploredtest/gymjs';

import { OnPolicyAlgorithm } from '../common/on_policy_algorithm';
import { clipGradNorm, explainedVariance } from '../common/utils';
import {
  A2CHyperParameters,
  ActorCriticParams,
  RolloutBufferResult,
} from '../common/type_aliases';
import { BaseCallback } from '../common/callbacks';

const DEFAULT_HYPER_PARAMETERS: A2CHyperParameters = {
  nSteps: 5,
  gamma: 0.99,
  gaeLmbda: 1.0,
  entCoef: 0.0,
  vfCoef: 0.5,
  maxGradNorm: 0.5,
  normalizeadvantage: true,
};

const DEFAULT_POLICY_PARAMETERS: ActorCriticParams = {
  lr: 3e-4,
  activation: 'relu',
  optim: undefined,
  policyNet: [64, 64],
  valueNet: [64, 64],
  logStdInit: 0.0,
};

export class A2C<
  ObsType extends tfCore.Tensor,
  ActType extends tfCore.Tensor | number,
> extends OnPolicyAlgorithm<ObsType, ActType> {
  public normalizeadvantage: boolean;

  constructor(
    env: gym.Env<ObsType, ActType>,
    hyperParameters: Partial<A2CHyperParameters> = {},
    policyParameters: Partial<ActorCriticParams> = {},
    verbose = false
  ) {
    const fHyperParameters = Object.assign(
      {},
      DEFAULT_HYPER_PARAMETERS,
      hyperParameters
    ) as A2CHyperParameters;
    const fPolicyParameters = Object.assign(
      {},
      DEFAULT_POLICY_PARAMETERS,
      policyParameters
    ) as ActorCriticParams;
    super(env, fHyperParameters, fPolicyParameters, verbose);
    this.normalizeadvantage = fHyperParameters.normalizeadvantage;
  }

  async train(): Promise<void> {
    let entropyLosses: number[] = [];
    let pgLosses: number[] = [];
    let valueLosses: number[] = [];
    let lossValue: number = 0;

    for (let batch of this.buffer.get(this.buffer.size)) {
      let [observations, actions, oldValues, oldLogProb, advantages, returns] =
        batch;

      if (this.normalizeadvantage && advantages.size > 1) {
        advantages = tf.tidy(() => {
          const { mean, variance } = tf.moments(advantages);
          const std = variance.sqrt();
          return advantages.sub(mean).div(std.add(tf.scalar(1e-8)));
        });
      }

      const optFunc = () => {
        const [values, logProb, entropy] = this.policy.evaluateActions(
          observations,
          actions
        );

        const policyLoss = tf.neg(advantages.mul(logProb)).mean();
        pgLosses.push(policyLoss.asScalar().dataSync()[0]);

        const valueLoss = returns.sub(values).square().mean();
        valueLosses.push(valueLoss.asScalar().dataSync()[0]);

        const entropyLoss = tf.neg(entropy.mean());
        entropyLosses.push(entropyLoss.asScalar().dataSync()[0]);

        const loss = policyLoss
          .add(entropyLoss.mul(this.entCoef))
          .add(valueLoss.mul(this.vfCoef));
        lossValue = loss.asScalar().dataSync()[0];

        return loss.asScalar();
      };

      tf.tidy(() => {
        const result = this.policy.optim.computeGradients(optFunc);
        const normedGrad: tfCore.NamedTensorMap = clipGradNorm(
          result.grads,
          this.maxGradNorm
        );
        this.policy.optim.applyGradients(normedGrad);
      });
    }

    const explainedVar = explainedVariance(
      (this.buffer.stackedTensors as RolloutBufferResult)['values'],
      this.buffer.returns
    );

    this.logger.record(
      'Entropy Loss',
      tf.util.sum(entropyLosses) / entropyLosses.length
    );
    this.logger.record(
      'Policy Gradient Loss',
      tf.util.sum(pgLosses) / pgLosses.length
    );
    this.logger.record(
      'Value Loss',
      tf.util.sum(valueLosses) / valueLosses.length
    );
    this.logger.record('Loss', lossValue);
    this.logger.record('Explained Variance', explainedVar.dataSync()[0]);

    const logStd = this.policy.getLogStd();
    if (logStd) {
      this.logger.record('Log STD', tf.exp(logStd).mean().dataSync()[0]);
    }

    tf.dispose(explainedVar);
  }

  async learn(
    totalTimesteps: number,
    callBack: BaseCallback<ObsType, ActType> | null,
    resetNumTimesteps: boolean = false,
    logInterval: number | null = 100
  ): Promise<void> {
    await super.learn(totalTimesteps, callBack, resetNumTimesteps, logInterval);
  }
}
