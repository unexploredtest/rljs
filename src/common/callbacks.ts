import * as tf from "@tensorflow/tfjs";
import * as gym from "@unexploredtest/gymjs";
import { BaseAlgorithm } from "./base_algorithm";
import { Logger } from "./logger";

export abstract class BaseCallback<
  ObsType extends tf.Tensor,
  ActType extends number | tf.Tensor,
> {
  protected model: BaseAlgorithm<ObsType, ActType> | null;
  protected nCalls: number;
  protected numTimesteps: number;
  protected verbose: boolean;

  constructor(verbose: boolean) {
    this.model = null;
    this.nCalls = 0;
    this.numTimesteps = 0;
    this.verbose = verbose;
  }

  get env(): gym.Env<ObsType, ActType> {
    if (this.model === null) {
      throw new Error("Model is null, can't get env. Init callback first.");
    }
    return this.model.env;
  }

  get logger(): Logger {
    if (this.model === null) {
      throw new Error("Model is null, can't get logger. Init callback first.");
    }
    return this.model.logger;
  }

  abstract _initCallback(): void | Promise<void>;

  async initCallback(model: BaseAlgorithm<ObsType, ActType>): Promise<void> {
    this.model = model;
    const result = this._initCallback();
    if (result instanceof Promise) {
      await result;
    }
  }

  abstract _onTrainingStart(): void | Promise<void>;

  async onTrainingStart(): Promise<void> {
    if (this.model === null) {
      throw new Error(
        "Model is null, can't start training. Init callback first.",
      );
    }
    this.numTimesteps = this.model.numTimesteps;
    const result = this._onTrainingStart();
    if (result instanceof Promise) {
      await result;
    }
  }

  abstract _onRolloutStart(): void | Promise<void>;

  async onRolloutStart(): Promise<void> {
    const result = this._onRolloutStart();
    if (result instanceof Promise) {
      await result;
    }
  }

  abstract _onStep(): boolean | Promise<boolean>;

  async onStep(): Promise<boolean> {
    if (this.model === null) {
      throw new Error(
        "Model is null, can't start training. Init callback first.",
      );
    }

    this.numTimesteps = this.model.numTimesteps;
    this.nCalls += 1;
    const result = this._onStep();
    if (result instanceof Promise) {
      return await result;
    }
    return result;
  }

  abstract _onTrainingEnd(): void | Promise<void>;

  async onTrainingEnd(): Promise<void> {
    const result = this._onTrainingEnd();
    if (result instanceof Promise) {
      await result;
    }
  }

  abstract _onRolloutEnd(): void | Promise<void>;

  async onRolloutEnd(): Promise<void> {
    const result = this._onRolloutEnd();
    if (result instanceof Promise) {
      await result;
    }
  }
}
