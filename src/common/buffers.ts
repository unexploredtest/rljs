import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces'

import { RolloutBufferSamples } from './type_aliases';

abstract class BaseBuffer {
    protected bufferSize: number;
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;
    // private actionDim: number[];
    protected pos: number;
    protected full: boolean;

    constructor(bufferSize: number, observationSpace: spaces.Space, actionSpace: spaces.Space) {
        this.bufferSize = bufferSize;
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;

        this.pos = 0;
        this.full = false;
    }

    get size(): number {
        if(this.full) {
            return this.pos
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
        const upperBound = this.full ? this.bufferSize : this.pos;
        const batchInds = tf.randomUniformInt([batchSize], 0, upperBound);
        return this.getSamples(batchInds);
    }

    protected abstract getSamples(indices: tf.Tensor): tf.Tensor[];
}

export class RolloutBuffer extends BaseBuffer {
    observations: tf.Tensor;
    actions: tf.Tensor;
    rewards: tf.Tensor;
    returns: tf.Tensor;
    episodeStarts: tf.Tensor;
    logProbs: tf.Tensor;
    values: tf.Tensor;
    advantages: tf.Tensor;

    gaeLambda: number;
    gamma: number;

    constructor(bufferSize: number, observationSpace: spaces.Space, actionSpace: spaces.Space, gaeLambda: number, gamma: number) {
        super(bufferSize, observationSpace, actionSpace);
        this.gaeLambda = gaeLambda;
        this.gamma = gamma;
        this.observations = tf.zeros([this.bufferSize, ...this.observationSpace.shape], this.observationSpace.dtype);
        this.actions = tf.zeros([this.bufferSize, ...this.actionSpace.shape], this.actionSpace.dtype);
        this.rewards = tf.zeros([this.bufferSize], "float32");
        this.returns = tf.zeros([this.bufferSize], "float32");
        this.episodeStarts = tf.zeros([this.bufferSize], "float32");
        this.values = tf.zeros([this.bufferSize], "float32");
        this.logProbs = tf.zeros([this.bufferSize], "float32");
        this.advantages = tf.zeros([this.bufferSize], "float32");
        this.reset();
    }

    reset(): void {
        this.observations = tf.zeros([this.bufferSize, ...this.observationSpace.shape], this.observationSpace.dtype);
        this.actions = tf.zeros([this.bufferSize, ...this.actionSpace.shape], this.actionSpace.dtype);
        this.rewards = tf.zeros([this.bufferSize], "float32");
        this.returns = tf.zeros([this.bufferSize], "float32");
        this.episodeStarts = tf.zeros([this.bufferSize], "float32");
        this.values = tf.zeros([this.bufferSize], "float32");
        this.logProbs = tf.zeros([this.bufferSize], "float32");
        this.advantages = tf.zeros([this.bufferSize], "float32");
        super.reset();
    }

    add(obs: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, episodeStart: tf.Tensor, value: tf.Tensor, logProb: tf.Tensor): void {
        this.observations = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([obs]));
        this.actions = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([action]));
        this.rewards = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([reward]));
        this.episodeStarts = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([episodeStart]));
        this.values = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([value]));
        this.logProbs = tf.tensorScatterUpdate(this.observations, tf.tensor1d([this.pos]), tf.stack([logProb]));

        this.pos += 1;
        if(this.pos == this.bufferSize) {
            this.full = true;
        }
    }

    protected getSamples(indices: tf.Tensor): RolloutBufferSamples {
        let data = [
            this.observations.gather(indices),
            this.actions.gather(indices),
            this.values.gather(indices),
            this.logProbs.gather(indices),
            this.advantages.gather(indices),
            this.returns.gather(indices)
        ] as RolloutBufferSamples;

        return data;
    }

    computeReturnsAndAdvantages(lastValues: tf.Tensor, dones: tf.Tensor): void {
        lastValues = lastValues.flatten();

        let lastGaeLam = tf.zeros(dones.shape);
        for(let step = this.bufferSize - 1; step >= 0; step--) {
            let nextNonTerminal: tf.Tensor;
            let nextValues: tf.Tensor;
            if(step == this.bufferSize - 1) {
                nextNonTerminal = tf.scalar(1).sub(dones.asType("float32"));
                nextValues = lastValues;
            } else {
                nextNonTerminal = tf.scalar(1).sub(this.episodeStarts.gather(step + 1));
                nextValues = this.values.gather(step + 1);
            }

            // Don't blame me for this ugly line of code, blame js/ts for not supporting operator overloading
            let delta = this.rewards.gather(step).add(tf.scalar(this.gamma).mul(nextValues).mul(nextNonTerminal)).sub(this.values.gather(step));
            lastGaeLam = delta.add(tf.scalar(this.gamma * this.gaeLambda).mul(nextNonTerminal).mul(lastGaeLam))

            this.advantages = tf.tensorScatterUpdate(this.advantages, tf.tensor1d([step]), tf.stack([lastGaeLam]));
        }

        this.returns = this.advantages.add(this.values);
    }

}