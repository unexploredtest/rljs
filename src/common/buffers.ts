import * as tf from "@tensorflow/tfjs";
import * as gym from "gymjs";

import { RolloutBufferResult, RolloutBufferSamples } from "./type_aliases";

abstract class BaseBuffer {
  protected bufferSize: number;
  protected observationShape: number[];
  protected actionShape: number[];
  // private actionDim: number[];
  protected pos: number;
  protected full: boolean;

  constructor(
    bufferSize: number,
    observationShape: number[],
    actionShape: number[],
  ) {
    this.bufferSize = bufferSize;
    this.observationShape = observationShape;
    this.actionShape = actionShape;

    this.pos = 0;
    this.full = false;
  }

  get size(): number {
    if (this.full) {
      return this.pos;
    } else {
      return this.bufferSize;
    }
  }

  abstract add(...args: tf.Tensor[]): void;

  extend(...args: tf.Tensor[][]): void {
    for (let tensorArray of args) {
      this.add(...tensorArray);
    }
  }

  reset(): void {
    this.pos = 0;
    this.full = false;
  }

  sample(batchSize: number): tf.Tensor[] {
    return tf.tidy(() => {
      const upperBound = this.full ? this.bufferSize : this.pos;
      const batchInds = tf.randomUniformInt([batchSize], 0, upperBound);
      return this.getSamples(batchInds);
    });
  }

  protected abstract getSamples(indices: tf.Tensor): tf.Tensor[];
}

export class RolloutBuffer extends BaseBuffer {
  observations: tf.Tensor[];
  actions: tf.Tensor[];
  rewards: tf.Tensor[];
  returns: tf.Tensor;
  episodeStarts: tf.Tensor[];
  logProbs: tf.Tensor[];
  values: tf.Tensor[];
  advantages: tf.Tensor[];
  stackedTensors: RolloutBufferResult | null;

  gaeLambda: number;
  gamma: number;

  constructor(
    bufferSize: number,
    observationShape: number[],
    actionShape: number[],
    gaeLambda: number,
    gamma: number,
  ) {
    super(bufferSize, observationShape, actionShape);
    this.gaeLambda = gaeLambda;
    this.gamma = gamma;
    // Unfortunately I can't just call this.reset() and expect TS to figure it out
    this.observations = [];
    this.actions = [];
    this.rewards = [];
    this.returns = tf.zeros([bufferSize]);
    this.episodeStarts = [];
    this.values = [];
    this.logProbs = [];
    this.advantages = [];
    this.stackedTensors = null;
    super.reset();
  }

  reset(): void {
    tf.dispose([
      this.observations,
      this.actions,
      this.returns,
      this.episodeStarts,
      this.values,
      this.logProbs,
      this.advantages,
    ]);

    if (this.stackedTensors !== null) {
      tf.dispose(this.stackedTensors);
    }

    this.observations = [];
    this.actions = [];
    this.rewards = [];
    this.returns = tf.zeros([this.bufferSize]);
    this.episodeStarts = [];
    this.values = [];
    this.logProbs = [];
    this.advantages = [];
    this.stackedTensors = null;
    super.reset();
  }

  add(
    obs: tf.Tensor,
    action: tf.Tensor,
    reward: tf.Tensor,
    episodeStart: tf.Tensor,
    value: tf.Tensor,
    logProb: tf.Tensor,
  ): void {
    this.observations.push(obs);
    this.actions.push(action);
    this.rewards.push(reward);
    this.episodeStarts.push(episodeStart);
    this.values.push(value);
    this.logProbs.push(logProb);

    this.pos += 1;
    if (this.pos == this.bufferSize) {
      this.full = true;
    }
  }

  *get(batchSize: number): Generator<RolloutBufferSamples> {
    let indicesArray = Array.from(Array(this.pos).keys());
    tf.util.shuffle(indicesArray);
    const indices = tf.tensor(indicesArray, [this.pos], "int32");
    for (let startIdx = 0; startIdx < this.pos; startIdx += batchSize) {
      let endIndex = startIdx + batchSize;
      if (endIndex >= this.pos) {
        endIndex = this.pos - 1;
      }

      yield this.getSamples(
        indices.gather(tf.range(startIdx, endIndex, 1, "int32")),
      );
    }
    tf.dispose(indices);
  }

  protected getSamples(indices: tf.Tensor): RolloutBufferSamples {
    if (this.stackedTensors === null) {
      this.computeStackedTensors();
    }

    return tf.tidy(() => {
      if (this.stackedTensors === null) {
        throw new Error(
          "Stacked tensors shouldn't be null, something must have gone wrong.",
        );
      }

      const data = [
        this.stackedTensors["observations"].gather(indices),
        this.stackedTensors["actions"].gather(indices),
        this.stackedTensors["values"].gather(indices),
        this.stackedTensors["logProb"].gather(indices),
        this.stackedTensors["advantages"].gather(indices),
        this.returns.gather(indices),
      ] as RolloutBufferSamples;

      return data;
    });
  }

  computeStackedTensors() {
    if (this.stackedTensors !== null) {
      tf.dispose(this.stackedTensors);
    }

    this.stackedTensors = {
      observations: tf
        .stack(this.observations)
        .reshape([this.pos, ...this.observationShape]),
      actions: tf.stack(this.actions).reshape([this.pos, ...this.actionShape]),
      values: tf.stack(this.values).reshape([this.pos]),
      logProb: tf.stack(this.logProbs).reshape([this.pos]),
      advantages: tf.stack(this.advantages).reshape([this.pos]),
    };
  }

  computeReturnsAndAdvantages(lastValues: tf.Tensor, dones: tf.Tensor): void {
    lastValues = lastValues.flatten();

    let lastGaeLam = tf.zeros(dones.shape);
    for (let step = this.bufferSize - 1; step >= 0; step--) {
      let nextNonTerminal: tf.Tensor;
      let nextValues: tf.Tensor;
      if (step == this.bufferSize - 1) {
        nextNonTerminal = tf.scalar(1).sub(dones.asType("float32"));
        nextValues = lastValues;
      } else {
        nextNonTerminal = tf.scalar(1).sub(this.episodeStarts[step + 1]);
        nextValues = this.values[step + 1];
      }

      // Don't blame me for this ugly line of code, blame js/ts for not supporting operator overloading
      let delta = this.rewards[step]
        .add(tf.scalar(this.gamma).mul(nextValues).mul(nextNonTerminal))
        .sub(this.values[step]);
      lastGaeLam = delta.add(
        tf
          .scalar(this.gamma * this.gaeLambda)
          .mul(nextNonTerminal)
          .mul(lastGaeLam),
      );
      this.advantages[step] = lastGaeLam;
      tf.dispose(nextNonTerminal);
      // tf.dispose(nextValues);
      tf.dispose(delta);
    }
    // tf.dispose(lastGaeLam);
    this.returns = tf
      .stack(this.advantages)
      .add(tf.stack(this.values))
      .reshape([this.pos]);
  }
}
