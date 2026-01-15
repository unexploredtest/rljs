import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';
import * as gym from '@unexploredtest/gymjs';

import { OnPolicyAlgorithm } from '../common/on_policy_algorithm';
import { clipGradNorm, explainedVariance, tensorToCore, tensorToTf } from '../common/utils';
import {
  DQNHyperParams,
  QNetworkParams,
  RolloutBufferResult,
} from '../common/type_aliases';
import { BaseCallback } from '../common/callbacks';
import { OffPolicyAlgorithm } from '../common/off_policy_algorithm';
import { DQNPolicy } from '../common/policies';

const DEFAULT_HYPER_PARAMETERS: DQNHyperParams = {
  bufferSize: 100_000,
  learningStarts: 100,
  batchSize: 32,
  tau: 1.0,
  gamma: 0.99,
  trainFreq: 4,
  gradientSteps: 1,
  nSteps: 1,
  maxGradNorm: 10,
  initialEps: 1.0,
  finalEps: 0.05,
  explorationFraction: 0.1,
  targetUpdateInterval: 10000,
};

const DEFAULT_POLICY_PARAMETERS: QNetworkParams = {
  lr: 1e-4,
  activation: 'relu',
  optim: undefined,
  net: [64, 64],
};

export class DQN<
  ObsType extends tfCore.Tensor,
  ActType extends tfCore.Tensor | number,
> extends OffPolicyAlgorithm<ObsType, ActType> {
  public policy: DQNPolicy<ObsType, ActType>;

  public maxGradNorm: number;
  public initialEps: number;
  public finalEps: number;
  public explorationFraction: number;
  public targetUpdateInterval: number;
  public explorationRate: number;
  public nCalls: number;

  constructor(
    env: gym.Env<ObsType, ActType>,
    hyperParameters: Partial<DQNHyperParams> = {},
    policyParameters: Partial<QNetworkParams> = {},
    verbose = false
  ) {
    const fHyperParameters = Object.assign(
      {},
      DEFAULT_HYPER_PARAMETERS,
      hyperParameters
    ) as DQNHyperParams;
    const fPolicyParameters = Object.assign(
      {},
      DEFAULT_POLICY_PARAMETERS,
      policyParameters
    ) as QNetworkParams;

    super(env, fHyperParameters, fPolicyParameters, verbose);

    this.policy = new DQNPolicy(
      this.observationSpace,
      this.actionSpace,
      fPolicyParameters
    );

    this.maxGradNorm = fHyperParameters.maxGradNorm;
    this.initialEps = fHyperParameters.initialEps;
    this.finalEps = fHyperParameters.finalEps;
    this.explorationFraction = fHyperParameters.explorationFraction;
    this.targetUpdateInterval = fHyperParameters.targetUpdateInterval;
    this.explorationRate = 0.0;
    this.nCalls = 0;
  }

  async train(): Promise<void> {
    const losses: number[] = [];

    for (let i = 0; i < this.gradientStepsUse; i++) {
      const [
        observations,
        nextObservations,
        actions,
        rewards,
        dones,
        timeouts,
      ] = this.buffer.sample(this.batchSize);

      const nextQValuesRaw = this.policy.qNetTarget.forward(
        nextObservations,
        true
      );
      const nextQValues = nextQValuesRaw.max(1).reshape([-1, 1]);
      const targetQValues = rewards.add(
        tf.scalar(1).sub(dones).mul(this.gamma).mul(nextQValues)
      );

      const optFunc = () => {
        const currentQValuesRaw = this.policy.qNet.forward(observations, true);
        const currentQValues = tf.gather(
          currentQValuesRaw,
          actions.asType('int32'),
          1,
          1
        );

        // TODO: Use Huber loss instead of MSE like stable baselines 3
        const loss = currentQValues.sub(targetQValues).square().mean();
        losses.push(loss.asScalar().dataSync()[0]);

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

    this.nUpdates += this.gradientStepsUse;

    this.logger.record('nUpdates', this.nUpdates);
    this.logger.record('Loss', tf.util.sum(losses) / losses.length);
  }

  // Overrides the base_class predict function to include epsilon-greedy exploration.
  override predict(obs: ObsType, deterministic: boolean): tfCore.Tensor {
    if (this.policy === null) {
      throw new Error('Policy not defined!');
    }

    if (!deterministic && Math.random() < this.explorationRate) {
      return tf.tensor([this.actionSpace.sample()]);
    } else {
      if (obs instanceof tf.Tensor) {
        return tensorToCore(this.policy.predict(tensorToTf(obs), deterministic));
      } else {
        return tensorToCore(this.policy.predict(obs, deterministic));
      }
    }
  }

  protected override onStep() {
    this.nCalls++;

    // Update target network
    if (this.nCalls % this.targetUpdateInterval === 0) {
      tf.tidy(() => {
        const orgParams = this.policy.qNet.model.getWeights();
        const targetParams = this.policy.qNet.model.getWeights();
        const newParams = targetParams.map((value, index) => {
          return value.mul(1 - this.tau).add(orgParams[index].mul(this.tau));
        });
        this.policy.qNetTarget.model.setWeights(newParams);
      });
    }

    // Update exploration rate
    const progressRemaining = 1 - this.numTimesteps / this.totalTimesteps;

    if (1 - progressRemaining > this.explorationFraction) {
      this.explorationRate = this.finalEps;
    } else {
      this.explorationRate =
        this.initialEps +
        ((1 - progressRemaining) * (this.finalEps - this.initialEps)) /
          this.explorationFraction;
    }

    this.logger.recordMean('Exploration Rate', this.explorationRate);
  }

  async learn(
    totalTimesteps: number,
    callBack: BaseCallback<ObsType, ActType> | null = null,
    resetNumTimesteps: boolean = false,
    logInterval: number | null = 4
  ): Promise<void> {
    await super.learn(totalTimesteps, callBack, resetNumTimesteps, logInterval);
  }

  async saveModels(modelURL: string): Promise<void> {
    await this.policy.saveModels(modelURL);
  }

  async loadModels(modelURL: string): Promise<void> {
    await this.policy.loadModels(modelURL);
  }
}
