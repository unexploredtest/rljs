import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';

import { BasePolicy, ActorCriticPolicy } from './policies.js';
import { BaseHyperParams } from './type_aliases.js';
import { Logger } from './logger.js';
import { BaseCallback } from './callbacks.js';


export abstract class BaseAlgorithm<ObsType extends tf.Tensor, ActType extends tf.Tensor | number> {
    public env: gym.Env<ObsType, ActType>;
    public observationSpace: gym.spaces.Space<ObsType>;
    public actionSpace: gym.spaces.Space<ActType>;
    public numTimesteps: number;
    public numTimestepsAtStart: number;
    public policy: BasePolicy<ObsType, ActType> | null;
    public lastObs: tf.Tensor | null;
    public lastEpisodeStarts: tf.Tensor | null;
    public logger: Logger;
    public startTime: number;
    protected verbose: boolean;
    constructor(env: gym.Env<ObsType, ActType>, hyperParams: BaseHyperParams, verbose: boolean) {
        this.env = env;
        this.observationSpace = env.observationSpace;
        this.actionSpace = env.actionSpace;
        this.lastEpisodeStarts = null;
        this.policy = null;

        this.lastObs = null;

        this.verbose = verbose;
        this.numTimesteps = 0;
        this.numTimestepsAtStart = 0;
        this.startTime = Date.now();
        this.logger = new Logger();
    }

    private initCallback(maybeCallback: BaseCallback<ObsType, ActType> | null): BaseCallback<ObsType, ActType> | null {
        maybeCallback?.initCallback(this);
        return maybeCallback;
    }

    protected setupLearn(totalTimesteps: number, callBack: BaseCallback<ObsType, ActType> | null, resetNumTimesteps: boolean = false): [number, BaseCallback<ObsType, ActType> | null] {
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
        const callBackFinal = this.initCallback(callBack);
        return [totalTimesteps, callBackFinal];
    }

    abstract learn(totalTimesteps: number, callBack: BaseCallback<ObsType, ActType> | null, resetNumTimesteps: boolean, logInterval: number | null): void;

    predict(obs: ObsType, deterministic: boolean) {
        if(this.policy === null) {
            throw new Error("Policy not defined!");
        }
        return this.policy.predict(obs, deterministic);
    }

    abstract saveModels(...URLs: string[]): Promise<void>;
    abstract loadModels(...URLs: string[]): Promise<void>;
}


