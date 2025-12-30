import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';

import { BasePolicy, ActorCriticPolicy } from './policies.js';
import { BaseHyperParams } from './type_aliases.js';
import { Logger } from './logger.js';


export abstract class BaseAlgorithm<ObsType extends tf.Tensor, ActType extends tf.Tensor | number> {
    public env: gym.Env<ObsType, ActType>;
    protected observationSpace: gym.spaces.Space<ObsType>;
    protected actionSpace: gym.spaces.Space<ActType>;
    protected numTimesteps: number;
    protected numTimestepsAtStart: number;
    public policy: BasePolicy<ObsType, ActType> | null;
    protected lastObs: tf.Tensor | null;
    protected lastEpisodeStarts: tf.Tensor | null;
    protected logger: Logger;
    protected startTime: number;
    constructor(env: gym.Env<ObsType, ActType>, hyperParams: BaseHyperParams) {
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
            let [obs, _] = this.env.reset();
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

    predict(obs: ObsType, deterministic: boolean) {
        if(this.policy === null) {
            throw new Error("Policy not defined!");
        }
        return this.policy.predict(obs, deterministic);
    }

    abstract saveModels(...URLs: string[]): Promise<void>;
    abstract loadModels(...URLs: string[]): Promise<void>;
}


