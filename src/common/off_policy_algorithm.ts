import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';
import * as gym from '@unexploredtest/gymjs';

import { BaseAlgorithm } from './base_algorithm';
import { ReplayBuffer } from './buffers';
import { ActorCriticPolicy, BasePolicy } from './policies';
import { BaseModelParams, OffPolicyHyperParams } from './type_aliases';
import { ActorCriticParams } from './type_aliases';
import { clipTensor, tensorToCore, tensorToTf } from './utils';
import { BaseCallback } from './callbacks';

export abstract class OffPolicyAlgorithm<
  ObsType extends tfCore.Tensor,
  ActType extends tfCore.Tensor | number,
> extends BaseAlgorithm<ObsType, ActType> {
  public buffer: ReplayBuffer;
  public bufferSize: number;
  public learningStarts: number;
  public batchSize: number;
  public tau: number;
  public gamma: number;
  public trainFreq: number;
  public gradientSteps: number;
  public gradientStepsUse: number;
  public nSteps: number;
  public policy: BasePolicy<ObsType, ActType> | null;
  private totalReward: number;
  private episodeLength: number;

  constructor(
    env: gym.Env<ObsType, ActType>,
    hyperParams: OffPolicyHyperParams,
    policyParams: BaseModelParams,
    verbose: boolean
  ) {
    super(env, hyperParams, verbose);
    this.bufferSize = hyperParams.bufferSize;
    this.learningStarts = hyperParams.learningStarts;
    this.batchSize = hyperParams.batchSize;
    this.tau = hyperParams.tau;
    this.gamma = hyperParams.gamma;
    this.trainFreq = hyperParams.trainFreq;
    this.gradientSteps = hyperParams.gradientSteps;
    this.gradientStepsUse = hyperParams.gradientSteps;
    this.nSteps = hyperParams.nSteps;

    this.buffer = new ReplayBuffer(
      this.bufferSize,
      this.observationSpace.shape,
      this.actionSpace.shape
    );

    this.totalReward = 0;
    this.episodeLength = 0;
    this.policy = null;
  }

  abstract train(): Promise<void>;

  private storeTransition(
    bufferAction: tfCore.Tensor,
    newObs: tfCore.Tensor,
    reward: tfCore.Tensor,
    dones: tfCore.Tensor,
    truns: tfCore.Tensor
  ) {
    if (this.lastObs === null) {
      throw new Error("Last Obs can't be null! Reset the environment first.");
    }

    this.buffer.add(this.lastObs, newObs, bufferAction, reward, dones, truns);
  }

  private sampleAction(): [tfCore.Tensor, tfCore.Tensor] {
    if (this.policy === null) {
      throw new Error('Policy should be defined.');
    }

    if (this.lastObs === null) {
      throw new Error('Last obs must not be null!');
    }

    let unscaledAction: tfCore.Tensor;
    let action: tfCore.Tensor;

    if (this.numTimesteps < this.learningStarts) {
      const action = this.actionSpace.sample();
      if (!(action instanceof tf.Tensor)) {
        unscaledAction = tf.tensor([action]);
      } else {
        unscaledAction = action;
      }
    } else {
      unscaledAction = this.predict(
        this.lastObs.reshape([-1, ...this.lastObs.shape]),
        false
      ).reshape([-1]);
    }

    action = unscaledAction.clone();

    return [action, unscaledAction];
  }

  async collectRollouts(
    callBack: BaseCallback<ObsType, ActType> | null
  ): Promise<[number, number, boolean]> {
    if (this.lastObs === null) {
      throw new Error('Last obs must not be null!');
    }

    if (this.trainFreq < 1) {
      throw new Error('Train frequency must at least be 1');
    }

    let numCollectedSteps = 0;
    let numCollectedEpisodes = 0;

    await callBack?.onRolloutStart();

    while (numCollectedSteps < this.trainFreq) {
      let [actions, bufferActions] = this.sampleAction();

      let action: tfCore.Tensor | number;
      if (this.actionSpace instanceof gym.spaces.Discrete) {
        const actionData = await actions.asScalar().data();
        action = actionData[0];
      } else if (this.actionSpace instanceof gym.spaces.Box) {
        if (
          typeof this.actionSpace.low === 'number' &&
          typeof this.actionSpace.high === 'number'
        ) {
          action = actions.clipByValue(
            this.actionSpace.low,
            this.actionSpace.high
          );
        } else if (
          this.actionSpace.low instanceof tf.Tensor &&
          this.actionSpace.high instanceof tf.Tensor
        ) {
          action = clipTensor(
            actions,
            this.actionSpace.low,
            this.actionSpace.high
          );
        } else {
          throw new Error('Low and high must of the same type in Box!');
        }

        action = action.reshape(this.actionSpace.shape);
        action = tensorToCore(action);
      } else {
        throw new Error('Unknown action space');
      }

      let [obsRaw, reward, done, truncated, info] = await this.env.step(
        action as ActType
      );
      const obs = tensorToTf(obsRaw);
      tf.dispose(obsRaw);
      this.totalReward += reward;
      this.episodeLength++;

      this.numTimesteps++;
      numCollectedSteps++;

      const continueTraining = callBack?.onStep();
      if (callBack !== null && !continueTraining) {
        return [numCollectedSteps, numCollectedEpisodes, false];
      }

      this.storeTransition(
        bufferActions,
        obs,
        tf.tensor([reward]),
        tf.tensor([done]),
        tf.tensor([truncated])
      );

      this.onStep();

      this.lastObs = obs;

      if (done === true || truncated === true) {
        let [newObs, _] = this.env.reset();
        this.lastObs = newObs;

        this.totalRewards.push(this.totalReward);
        this.episodeLengths.push(this.episodeLength);

        this.totalReward = 0;
        this.episodeLength = 0;
      }
    }

    callBack?.onRolloutEnd();

    return [numCollectedSteps, numCollectedEpisodes, true];
  }

  protected onStep() {}

  async learn(
    totalTimesteps: number,
    callBack: BaseCallback<ObsType, ActType> | null,
    resetNumTimesteps: boolean = false,
    logInterval: number | null = 1
  ): Promise<void> {
    let iteration = 0;

    const callBackFinal = await this.setupLearn(
      totalTimesteps,
      callBack,
      resetNumTimesteps
    );

    await callBackFinal?.onTrainingStart();

    while (this.numTimesteps < this.totalTimesteps) {
      const [numCollectedSteps, numCollectedEpisodes, continueTraining] =
        await this.collectRollouts(callBackFinal);

      if (!continueTraining) {
        break;
      }

      iteration += 1;

      if (
        this.verbose &&
        logInterval !== null &&
        iteration % logInterval === 0
      ) {
        console.log(this.dumpLogs(iteration));
      }

      if (this.numTimesteps > 0 && this.numTimesteps > this.learningStarts) {
        this.gradientStepsUse =
          this.gradientSteps >= 0 ? this.gradientSteps : numCollectedEpisodes;
        if (this.gradientStepsUse > 0) {
          await this.train();
        }
      }
    }

    await callBackFinal?.onTrainingEnd();
  }

  dumpLogs(iteration: number = 0) {
    this.logger.record('Iteration', iteration);

    if (this.totalRewards.length > 0 && this.episodeLengths.length > 0) {
      this.logger.recordMean(
        'Reward Mean',
        tf.util.sum(this.totalRewards) / this.totalRewards.length
      );
      this.logger.recordMean(
        'Episode Length Mean',
        tf.util.sum(this.episodeLengths) / this.episodeLengths.length
      );
    }
    // Calculate the time difference in milliseconds
    const timeElapsed = (Date.now() - this.startTime) / 1000;
    const fps = (this.numTimesteps - this.numTimestepsAtStart) / timeElapsed;
    this.logger.record('FPS', Math.round(fps));
    this.logger.record('Time Elapsed', timeElapsed);
    return this.logger.dump(this.numTimesteps);
  }
}
