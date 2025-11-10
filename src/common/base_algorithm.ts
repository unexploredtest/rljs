import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces'

import { BasePolicy, ActorCriticPolicy } from './policies.js';


export abstract class BaseAlgorithm {
    public env: gym.Env | gym.Wrapper;
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;
    protected lr: number;
    protected numTimesteps: number;
    protected policy: ActorCriticPolicy;
    protected lastObs: tf.Tensor | null;
    protected lastEpisodeStarts: tf.Tensor| null;
    constructor(env: gym.Env | gym.Wrapper, lr: number) {
        this.env = env;
        this.observationSpace = env.observationSpace;
        this.actionSpace = env.actionSpace;
        this.lr = lr;
        this.lastEpisodeStarts = null;

        this.lastObs = null;

        this.policy = new ActorCriticPolicy("adam", this.observationSpace, this.actionSpace, lr);

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
        return this.policy.predict(obs, deterministic);
    }


}


