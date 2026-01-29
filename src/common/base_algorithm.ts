import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';
import * as gym from '@unexploredtest/gymjs';

import { BasePolicy, ActorCriticPolicy } from './policies';
import { BaseHyperParams } from './type_aliases';
import { Logger } from './logger';
import { BaseCallback } from './callbacks';
import { tensorToTf } from './utils';

export abstract class BaseAlgorithm<
  ObsType extends tfCore.Tensor,
  ActType extends tfCore.Tensor | number,
> {
  public env: gym.Env<ObsType, ActType>;
  public observationSpace: gym.spaces.Space<ObsType>;
  public actionSpace: gym.spaces.Space<ActType>;
  public totalTimesteps: number;
  public numTimesteps: number;
  public numTimestepsAtStart: number;
  public nUpdates: number;
  public policy: BasePolicy<ObsType, ActType> | null;
  public lastObs: tfCore.Tensor | null;
  public lastEpisodeStarts: tfCore.Tensor | null;
  public logger: Logger;
  public startTime: number;
  public totalRewards: number[];
  public episodeLengths: number[];
  protected verbose: boolean;
  constructor(
    env: gym.Env<ObsType, ActType>,
    hyperParams: BaseHyperParams,
    verbose: boolean
  ) {
    this.env = env;
    this.observationSpace = env.observationSpace;
    this.actionSpace = env.actionSpace;
    this.lastEpisodeStarts = null;
    this.policy = null;

    this.lastObs = null;

    this.verbose = verbose;
    this.totalTimesteps = 0;
    this.numTimesteps = 0;
    this.numTimestepsAtStart = 0;
    this.nUpdates = 0;
    this.totalRewards = [];
    this.episodeLengths = [];
    this.startTime = Date.now();
    this.logger = new Logger();
  }

  private async initCallback(
    maybeCallback: BaseCallback<ObsType, ActType> | null
  ): Promise<BaseCallback<ObsType, ActType> | null> {
    await maybeCallback?.initCallback(this);
    return maybeCallback;
  }

  protected async setupLearn(
    totalTimesteps: number,
    callBack: BaseCallback<ObsType, ActType> | null,
    resetNumTimesteps: boolean = false
  ): Promise<BaseCallback<ObsType, ActType> | null> {
    if (this.lastObs === null) {
      const [obsRaw, _] = this.env.reset();
      this.lastObs = tensorToTf(obsRaw);
      tf.dispose(obsRaw);
    }

    this.lastEpisodeStarts = tfCore.tensor([true], [1], 'float32');

    if (resetNumTimesteps) {
      this.numTimesteps = 0;
      this.totalTimesteps = totalTimesteps;
    } else {
      this.totalTimesteps = this.numTimesteps + totalTimesteps;
    }

    this.numTimestepsAtStart = this.numTimesteps;
    const callBackFinal = await this.initCallback(callBack);
    return callBackFinal;
  }

  abstract learn(
    totalTimesteps: number,
    callBack: BaseCallback<ObsType, ActType> | null,
    resetNumTimesteps: boolean,
    logInterval: number | null
  ): void;

  predict(obs: ObsType, deterministic: boolean) {
    if (this.policy === null) {
      throw new Error('Policy not defined!');
    }

    if (obs instanceof tfCore.Tensor) {
      return this.policy.predict(tensorToTf(obs), deterministic);
    } else {
      return this.policy.predict(obs, deterministic);
    }
  }

  abstract saveModels(...URLs: string[]): Promise<void>;
  abstract loadModels(...URLs: string[]): Promise<void>;
}
