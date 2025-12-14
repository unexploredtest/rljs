import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces'

import { BasePolicy, ActorCriticPolicy } from './policies.js';
import { BaseHyperParams } from './type_aliases.js';
import { Logger } from './logger.js';


export abstract class BaseAlgorithm {
    public env: gym.Env | gym.Wrapper;
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;
    protected numTimesteps: number;
    protected numTimestepsAtStart: number;
    protected policy: BasePolicy | null;
    protected lastObs: tf.Tensor | null;
    protected lastEpisodeStarts: tf.Tensor| null;
    protected logger: Logger;
    protected startTime: number;
    constructor(env: gym.Env | gym.Wrapper, hyperParams: BaseHyperParams) {
        this.env = env;
        this.observationSpace = env.observationSpace;
        this.actionSpace = env.actionSpace;
        this.lastEpisodeStarts = null;
        this.policy = null;

        this.lastObs = null;

        this.numTimesteps = 0;
        this.numTimestepsAtStart = 0;
        this.startTime = Date.now();
        this.logger = new Logger();
    }

    // abstract setupModel(): void;

    protected setupLearn(totalTimesteps: number, resetNumTimesteps: boolean = false): number {
        if(this.lastObs === null) {
            let [obs, _] = this.env.reset(undefined, null);
            this.lastObs = obs;
        }

        this.lastEpisodeStarts = tf.tensor([true], [1], "float32");

        if(resetNumTimesteps) {
            this.numTimesteps = 0;
        } else {
            totalTimesteps += this.numTimesteps;
        }

        this.numTimestepsAtStart = this.numTimesteps;

        return totalTimesteps;
    }

    abstract learn(totalTimesteps: number): void;

    predict(obs: tf.Tensor, deterministic: boolean) {
        if(this.policy === null) {
            throw new Error("Policy not defined!");
        }
        return this.policy.predict(obs, deterministic);
    }


}


