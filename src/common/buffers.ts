import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces'

import { RolloutBufferSamples } from './type_aliases.js';

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
        return tf.tidy(() => {
            const upperBound = this.full ? this.bufferSize : this.pos;
            const batchInds = tf.randomUniformInt([batchSize], 0, upperBound);
            return this.getSamples(batchInds);
        })
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

    gaeLambda: number;
    gamma: number;

    constructor(bufferSize: number, observationSpace: spaces.Space, actionSpace: spaces.Space, gaeLambda: number, gamma: number) {
        super(bufferSize, observationSpace, actionSpace);
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
        super.reset();
    }

    reset(): void {
        tf.dispose(this.observations);
        this.observations = [];
        tf.dispose(this.actions);
        this.actions = [];
        tf.dispose(this.returns);
        this.returns = tf.zeros([this.bufferSize]);
        tf.dispose(this.episodeStarts);
        this.episodeStarts = [];
        tf.dispose(this.values);
        this.values = [];
        tf.dispose(this.logProbs);
        this.logProbs = [];
        tf.dispose(this.advantages);
        this.advantages = [];
        // Fix action space later
        super.reset();
    }

    add(obs: tf.Tensor, action: tf.Tensor, reward: tf.Tensor, episodeStart: tf.Tensor, value: tf.Tensor, logProb: tf.Tensor): void {
        this.observations.push(obs);
        this.actions.push(action)
        this.rewards.push(reward);
        this.episodeStarts.push(episodeStart);
        this.values.push(value);
        this.logProbs.push(logProb);
        

        this.pos += 1;
        if(this.pos == this.bufferSize) {
            this.full = true;
        }
    }

    get(batchSize: number): RolloutBufferSamples[] {
        let indices = tf.range(0, this.pos, 1, "int32");
        let batches = [this.getSamples(indices)];
        tf.dispose(indices);
        // let indices = tf.randomUniformInt([this.bufferSize], 0, this.bufferSize - 1);
        // for(let startIdx = 0; startIdx < this.bufferSize; startIdx++) {
        //     let endIndex = startIdx + batchSize;
        //     if(endIndex >= this.bufferSize) {
        //         endIndex = this.bufferSize - 1;
        //     }

        //     batches.push(this.getSamples(indices));
        // }

        return batches;
    } 

    protected getSamples(indices: tf.Tensor): RolloutBufferSamples {
        let data = [
            tf.stack(this.observations).reshape([this.pos, ...this.observationSpace.shape]).gather(indices),
            tf.stack(this.actions).reshape([this.pos, ...this.actionSpace.shape]).gather(indices),
            tf.stack(this.values).reshape([this.pos]).gather(indices),
            tf.stack(this.logProbs).reshape([this.pos]).gather(indices),
            tf.stack(this.advantages).reshape([this.pos]).gather(indices),
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
                nextNonTerminal = tf.scalar(1).sub(this.episodeStarts[step + 1]);
                nextValues = this.values[step + 1];
            }

            // Don't blame me for this ugly line of code, blame js/ts for not supporting operator overloading
            let delta = this.rewards[step].add(tf.scalar(this.gamma).mul(nextValues).mul(nextNonTerminal)).sub(this.values[step]);
            lastGaeLam = delta.add(tf.scalar(this.gamma * this.gaeLambda).mul(nextNonTerminal).mul(lastGaeLam))
            this.advantages[step] = lastGaeLam;
            // tf.dispose(nextNonTerminal);
            // tf.dispose(nextValues);
            // tf.dispose(delta);
        }
        // tf.dispose(lastGaeLam);
        this.returns = tf.stack(this.advantages).add(tf.stack(this.values)).reshape([this.pos]);
    }
}