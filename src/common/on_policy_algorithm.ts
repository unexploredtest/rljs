import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';

import { BaseAlgorithm } from './base_algorithm';
import { RolloutBuffer } from './buffers';
import { ActorCriticPolicy } from './policies';
import { OnPolicyHyperParams } from './type_aliases';
import { ActorCriticParams } from './type_aliases';
import { clipTensor } from './utils';
import { BaseCallback } from './callbacks';


export abstract class OnPolicyAlgorithm<ObsType extends tf.Tensor, ActType extends tf.Tensor | number> extends BaseAlgorithm<ObsType, ActType> {
    public buffer: RolloutBuffer;
    public nSteps: number;
    public gamma: number;
    public gaeLmbda: number;
    public entCoef: number;
    public vfCoef: number;
    public maxGradNorm: number;
    public policy: ActorCriticPolicy<ObsType, ActType>;
    private totalReward: number;
    private episodeLength: number;

    constructor(env: gym.Env<ObsType, ActType>, hyperParams: OnPolicyHyperParams, policyParams: ActorCriticParams, verbose: boolean) {
        super(env, hyperParams, verbose);
        this.nSteps = hyperParams.nSteps;
        this.gamma = hyperParams.gamma;
        this.gaeLmbda = hyperParams.gaeLmbda;
        this.entCoef = hyperParams.entCoef;
        this.vfCoef = hyperParams.vfCoef;
        this.maxGradNorm = hyperParams.maxGradNorm;
        this.policy = new ActorCriticPolicy(this.observationSpace, this.actionSpace, policyParams);

        this.buffer = new RolloutBuffer(this.nSteps, this.observationSpace.shape, this.actionSpace.shape, this.gaeLmbda, this.gamma);

        this.totalReward = 0;
        this.episodeLength = 0;
    }

    abstract train(): Promise<void>;

    async collectRollouts(callBack: BaseCallback<ObsType, ActType> | null): Promise<boolean> {
        if(this.lastObs === null || this.lastEpisodeStarts === null) {
            throw new Error("Last obs/episode start must not be null!");
        }

        let curNSteps = 0;
        this.buffer.reset();
        this.lastObs;

        await callBack?.onRolloutStart();

        while(curNSteps < this.nSteps) {
            if(this.lastObs === null || this.lastEpisodeStarts === null) {
                throw new Error("Last obs/episode start must not be null!");
            }
            let [actions, values, logProbs] = this.policy.forward(this.lastObs, false);
            let action: tf.Tensor | number;
            if (this.actionSpace instanceof gym.spaces.Discrete) {
                const actionData = await actions.asScalar().data();
                action = actionData[0];
            } else if(this.actionSpace instanceof gym.spaces.Box) {
                // actions.print();
                if(typeof this.actionSpace.low === 'number' && typeof this.actionSpace.high === 'number') {
                    action = actions.clipByValue(this.actionSpace.low, this.actionSpace.high);
                } else if(this.actionSpace.low instanceof tf.Tensor && this.actionSpace.high instanceof tf.Tensor) {
                    action = clipTensor(actions, this.actionSpace.low, this.actionSpace.high);
                } else {
                    throw new Error('Low and high must of the same type in Box!');
                }

                action = action.reshape(this.actionSpace.shape);
            } else if(this.actionSpace instanceof gym.spaces.MultiDiscrete || this.actionSpace instanceof gym.spaces.MultiBinary) {
                action = actions.reshape(this.actionSpace.shape);
            } else {
                throw new Error('Unknown action space');
            }


            let [obs, reward, done, truncated, info] = await this.env.step(action as ActType);
            this.totalReward += reward;

            if(!done && truncated) {
                let values2 = this.policy.predictValues(obs);
                reward += this.gamma * values2.asScalar().dataSync()[0];
            }

            this.numTimesteps++;

            const continueTraining = callBack?.onStep();
            if(callBack !== null && !continueTraining) {
                return false;
            }

            this.buffer.add(this.lastObs, actions, tf.tensor([reward]), this.lastEpisodeStarts, values, logProbs);
            curNSteps++;

            // tf.dispose(this.lastObs);
            this.lastObs = obs;
            // tf.dispose(this.lastEpisodeStarts);
            this.lastEpisodeStarts = tf.tensor([done || truncated], [1], "float32");
            
            this.episodeLength += 1;

            if(done === true || truncated === true) {
                let [newObs, _] = this.env.reset();
                this.lastObs = newObs;

                this.logger.recordMean("Reward Mean", this.totalReward);
                this.logger.recordMean("Episode Length Mean", this.episodeLength);

                this.totalReward = 0;
                this.episodeLength = 0;
            }
            // tf.dispose(actions);
            // tf.dispose(values);
            // tf.dispose(logProbs);
            // tf.dispose(obs);
        }

        // console.log("avg: " + avg / i)

        const values = this.policy.predictValues(this.lastObs);
        this.buffer.computeReturnsAndAdvantages(values, this.lastEpisodeStarts);

        callBack?.onRolloutEnd();

        tf.dispose(values);

        return true;
    }

    async learn(totalTimesteps: number, callBack: BaseCallback<ObsType, ActType> | null, resetNumTimesteps: boolean = false, logInterval: number | null = 1): Promise<void> {
        let iteration = 0;

        const [totalTimestepsFinal, callBackFinal] = await this.setupLearn(totalTimesteps, callBack, resetNumTimesteps);

        await callBackFinal?.onTrainingStart();

        while(this.numTimesteps < totalTimestepsFinal) {
            const continueTraining = await this.collectRollouts(callBackFinal);

            if(!continueTraining) {
                break;
            }
            
            iteration += 1

            if(this.verbose && logInterval !== null && iteration % logInterval === 0) {
                console.log(this.dumpLogs(iteration));
            }
            
            await this.train();
            // console.log("Now: " + this.numTimesteps);
        }

        await callBackFinal?.onTrainingEnd();
    }

    dumpLogs(iteration: number = 0) {
        this.logger.record("Iteration", iteration);

        // Calculate the time difference in milliseconds
        const timeElapsed = (Date.now() - this.startTime) / 1000;
        const fps = (this.numTimesteps - this.numTimestepsAtStart) / timeElapsed
        this.logger.record("FPS", Math.round(fps));
        this.logger.record("Time Elapsed", timeElapsed);
        return this.logger.dump(this.numTimesteps);
    }

    async saveModels(PolicyURL: string, ValueURL: string): Promise<void> {
        await this.policy.saveModels(PolicyURL, ValueURL);
    }

    async loadModels(PolicyURL: string, ValueURL: string): Promise<void> {
        await this.policy.loadModels(PolicyURL, ValueURL);
    }
}