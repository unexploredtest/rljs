import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import * as spaces from 'gymjs/spaces';

import { BaseAlgorithm } from './base_algorithm.js';
import { RolloutBuffer } from './buffers.js';


export abstract class OnPolicyAlgorithm extends BaseAlgorithm {
    protected buffer: RolloutBuffer;
    protected nSteps: number;
    protected gamma: number;
    protected gaeLmbda: number;
    protected entCoef: number;
    protected vfCoef: number;
    protected maxGradNorm: number;

    constructor(env: gym.Env | gym.Wrapper, lr: number, nSteps: number, gamma: number, gaeLmbda: number, entCoef: number, vfCoef: number, maxGradNorm: number) {
        super(env, lr);
        this.nSteps = nSteps;
        this.gamma = gamma;
        this.gaeLmbda = gaeLmbda;
        this.entCoef = entCoef;
        this.vfCoef = vfCoef;
        this.maxGradNorm = maxGradNorm;

        this.buffer = new RolloutBuffer(nSteps, this.observationSpace, this.actionSpace, gaeLmbda, gamma);
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
            // tf.dispose(actions);
            // tf.dispose(values);
            // tf.dispose(logProbs);
            // tf.dispose(obs);
        }

        console.log("avg: " + avg / i)

        const values = this.policy.predictValues(this.lastObs);
        this.buffer.computeReturnsAndAdvantages(values, this.lastEpisodeStarts);

        return true;
    }

    async learn(totalTimesteps: number): Promise<void> {
        totalTimesteps = this.setupLearn(totalTimesteps);

        while(this.numTimesteps < totalTimesteps) {
            let continueTraining = await this.collectRollouts();

            await this.train();
            console.log("Now: " + this.numTimesteps);
        }
    }


}