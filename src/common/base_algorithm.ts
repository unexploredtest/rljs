import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces'

import { BasePolicy, ActorCriticPolicy } from './policies.js';
import { BaseHyperParams } from './type_aliases.js';


export abstract class BaseAlgorithm {
    public env: gym.Env | gym.Wrapper;
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;
    protected numTimesteps: number;
    protected policy: BasePolicy | null;
    protected lastObs: tf.Tensor | null;
    protected lastEpisodeStarts: tf.Tensor| null;
    constructor(env: gym.Env | gym.Wrapper, hyperParams: BaseHyperParams) {
        this.env = env;
        this.observationSpace = env.observationSpace;
        this.actionSpace = env.actionSpace;
        this.lastEpisodeStarts = null;
        this.policy = null;

        this.lastObs = null;

        this.numTimesteps = 0;
    }

    abstract setupModel(): void;

    protected setupLearn(totalTimesteps: number): number {
        if(this.lastObs === null) {
            let [obs, _] = this.env.reset();
            this.lastObs = obs;
        }

        this.lastEpisodeStarts = tf.tensor([true], [1], "float32");

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


