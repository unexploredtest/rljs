import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces';

import { BaseAlgorithm } from './base_algorithm.js';
import { RolloutBuffer } from './buffers.js';
import { ActorCriticPolicy } from './policies.js';
import { OnPolicyHyperParams } from './type_aliases.js';
import { ActorCriticParams } from './type_aliases.js';


export abstract class OnPolicyAlgorithm extends BaseAlgorithm {
    protected buffer: RolloutBuffer;
    protected nSteps: number;
    protected gamma: number;
    protected gaeLmbda: number;
    protected entCoef: number;
    protected vfCoef: number;
    protected maxGradNorm: number;
    protected policy: ActorCriticPolicy;

    constructor(env: gym.Env | gym.Wrapper, hyperParams: OnPolicyHyperParams, policyParams: ActorCriticParams) {
        super(env, hyperParams);
        this.nSteps = hyperParams.nSteps;
        this.gamma = hyperParams.gamma;
        this.gaeLmbda = hyperParams.gaeLmbda;
        this.entCoef = hyperParams.entCoef;
        this.vfCoef = hyperParams.vfCoef;
        this.maxGradNorm = hyperParams.maxGradNorm;
        this.policy = new ActorCriticPolicy(this.observationSpace, this.actionSpace, policyParams);

        this.buffer = new RolloutBuffer(this.nSteps, this.observationSpace, this.actionSpace, this.gaeLmbda, this.gamma);
    }

    setupModel(): void {

    }

    abstract train(): Promise<void>;

    async collectRollouts(): Promise<boolean> {
        if(this.lastObs === null || this.lastEpisodeStarts === null) {
            throw new Error("Last obs/episode start must not be null!");
        }

        let curNSteps = 0;
        this.buffer.reset();

        let avg = 0;
        let i = 0;
        let k = 0;

        while(curNSteps < this.nSteps) {
            k++;
            let [actions, values, logProbs] = this.policy.forward(this.lastObs, false);
            let actionData = await actions.asScalar().data();
            let action = actionData[0];
            let [obs, reward, done, truncated, info] = await this.env.step(action);

            this.numTimesteps++;
            curNSteps++;

            this.buffer.add(this.lastObs, tf.tensor([action], [1], "int32"), tf.tensor([reward]), this.lastEpisodeStarts, values, logProbs);

            // tf.dispose(this.lastObs);
            this.lastObs = obs;
            // tf.dispose(this.lastEpisodeStarts);
            this.lastEpisodeStarts = tf.tensor([done], [1], "float32");

            if(done === true) {
                avg += k;
                let [obs2, info2] = this.env.reset();
                this.lastObs = obs2;
                k = 0;
                i++;
            }
            tf.dispose(actions);
            // tf.dispose(values);
            // tf.dispose(logProbs);
            // tf.dispose(obs);
        }

        console.log("avg: " + avg / i)

        const values = this.policy.predictValues(this.lastObs);
        this.buffer.computeReturnsAndAdvantages(values, this.lastEpisodeStarts);
        tf.dispose(values);

        return true;
    }

    async learn(totalTimesteps: number): Promise<void> {
        totalTimesteps = this.setupLearn(totalTimesteps);

        while(this.numTimesteps < totalTimesteps) {
            console.time("exampleFunctionTime");
            let continueTraining = await this.collectRollouts();
            
            await this.train();
            console.log("Now: " + this.numTimesteps);
            console.timeEnd("exampleFunctionTime");
        }
    }


}