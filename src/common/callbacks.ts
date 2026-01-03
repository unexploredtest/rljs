import * as tf from '@tensorflow/tfjs';
import * as gym from 'gymjs';
import { BaseAlgorithm } from './base_algorithm';
import { Logger } from './logger';


export abstract class BaseCallback<ObsType extends tf.Tensor, ActType extends number | tf.Tensor> {
    protected model: BaseAlgorithm<ObsType, ActType> | null;
    protected nCalls: number;
    protected numTimesteps: number;
    protected verbose: boolean;

    constructor(verbose: boolean) {
        this.model = null;

        this.nCalls = 0;
        this.numTimesteps = 0;
        this.verbose = verbose;
    }

    get env(): gym.Env<ObsType, ActType> {
        if(this.model === null) {
            throw new Error("Model is null, can't get env. Init callback first.");
        }

        return this.model.env;
    }

    get logger(): Logger {
        if(this.model === null) {
            throw new Error("Model is null, can't get logger. Init callback first.");
        }

        return this.model.logger;
    }

    abstract _initCallback(): void;

    initCallback(model: BaseAlgorithm<ObsType, ActType>): void {
        this.model = model;
        this._initCallback();
    }

    abstract _onTrainingStart(): void;

    onTrainingStart(): void {
        if(this.model === null) {
            throw new Error("Model is null, can't start training. Init callback first.");
        }

        this.numTimesteps = this.model.numTimesteps;
        this._onTrainingStart();
    }

    abstract _onRolloutStart(): void;

    onRolloutStart(): void {
        this._onRolloutStart();
    }

    abstract _onStep(): boolean;

    onStep(): boolean {
        if(this.model === null) {
            throw new Error("Model is null, can't start training. Init callback first.");
        }

        this.numTimesteps = this.model.numTimesteps;
        this.nCalls += 1;

        return this._onStep();
    }

    abstract _onTrainingEnd(): void;

    onTrainingEnd(): void {
        this._onTrainingEnd();
    }

    abstract _onRolloutEnd(): void;

    onRolloutEnd(): void {
        this._onRolloutEnd();
    }
}