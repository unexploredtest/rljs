import * as tf from '@tensorflow/tfjs';
import * as spaces from 'gymjs/spaces'

import { CategoricalDistribution } from './distributions.js';
import { Distribution } from './distributions.js';

type Optimizer = "adam" | "SGD";

export abstract class baseModel {
    protected optimizer: Optimizer;
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;

    constructor(optimizer: Optimizer, observationSpace: spaces.Space, actionSpace: spaces.Space) {
        this.optimizer = optimizer;
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;
    }
}

export abstract class BasePolicy extends baseModel {
    abstract predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor;
}

export class ActorCriticPolicy extends BasePolicy {
    private actionDist: Distribution;
    public lr: number;
    public optim: tf.Optimizer;
    public policyModel: tf.LayersModel;
    public valueModel: tf.LayersModel;

    constructor(optimizer: Optimizer, observationSpace: spaces.Space, actionSpace: spaces.Space, lr: number, policyNet: number[], valueNet: number[]) {
        super(optimizer, observationSpace, actionSpace);
        this.lr = lr;
        if(actionSpace instanceof spaces.Discrete) {
            this.actionDist = new CategoricalDistribution((actionSpace as spaces.Discrete).n);
        } else {
            throw new Error("Not supported!");
        }

        this.optim = tf.train.adam(lr);

        const inputSize = this.observationSpace.shape;
        this.policyModel = this.createActor(policyNet);
        this.valueModel = this.createCritic(valueNet);
    }

    forward(obs: tf.Tensor, deterministic: boolean): [tf.Tensor, tf.Tensor, tf.Tensor] {
        return tf.tidy(() => { 
            if(obs.shape.length === 1) {
                obs = obs.reshape([-1, ...obs.shape])
            }
            // obs.print();
            let values = this.valueModel.predict(obs) as tf.Tensor; 
            let actionLogits = this.policyModel.predict(obs) as tf.Tensor;
            // actionLogits.print()
            this.actionDist.probaDistribution(actionLogits);
            let actions = this.actionDist.getActions(deterministic);
            let logProb = this.actionDist.logProb(actions);

            return [actions, values, logProb];
        })
    }

    predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
        return tf.tidy(() => { 
            if(obs.shape.length === 1) {
                obs = obs.reshape([-1, ...obs.shape])
            }
            let actionLogits = this.policyModel.predict(obs) as tf.Tensor;
            this.actionDist.probaDistribution(actionLogits);
            let actions = this.actionDist.getActions(deterministic);

            return actions;
        })
    }

    predictValues(obs: tf.Tensor): tf.Tensor {
        return tf.tidy(() => { 
            if(obs.shape.length === 1) {
                obs = obs.reshape([-1, ...obs.shape])
            }
            let values = this.valueModel.predict(obs) as tf.Tensor;
            return values;
        })
    }

    evaluateActions(obs: tf.Tensor, actions: tf.Tensor) {
        return tf.tidy(() => { 
            if(obs.shape.length === 1) {
                obs = obs.reshape([-1, ...obs.shape])
            }
            let values = this.valueModel.predict(obs) as tf.Tensor; 
            let actionLogits = this.policyModel.predict(obs) as tf.Tensor;
            this.actionDist.probaDistribution(actionLogits);
            let logProb = this.actionDist.logProb(actions);
            let entropy = this.actionDist.entropy();

            return [values, logProb, entropy];
        })
    }

    // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
    private createActor(netArch: number[]) {
        const input = tf.layers.input({shape: this.observationSpace.shape})
        let l = input
        netArch.forEach((units, i) => {
            l = tf.layers.dense({
                units, 
                activation: 'relu'
            }).apply(l) as tf.SymbolicTensor;
        })
        if (this.actionSpace instanceof spaces.Discrete) {
            l = tf.layers.dense({
                units: this.actionSpace.n, 
            }).apply(l) as tf.SymbolicTensor;
        } else {
            throw new Error('Unknown action space class');
        }
        return tf.model({inputs: input, outputs: l})
    }

    private createCritic(netArch: number[]) {
        const input = tf.layers.input({shape: this.observationSpace.shape})
        let l = input
        netArch.forEach((units, i) => {
            l = tf.layers.dense({
                units: units, 
                activation: 'relu'
            }).apply(l) as tf.SymbolicTensor;
        })
        l = tf.layers.dense({
            units: 1, 
            activation: 'linear'
        }).apply(l) as tf.SymbolicTensor;
        return tf.model({inputs: input, outputs: l})
    }
}

