import * as tf from '@tensorflow/tfjs';
import * as spaces from 'gymjs/spaces'

import { Distribution, CategoricalDistribution, DiagGaussianDistribution } from './distributions.js';
import { BaseModelParams, ActorCriticParams, ActivationType } from './type_aliases.js';

export abstract class baseModel {
    protected observationSpace: spaces.Space;
    protected actionSpace: spaces.Space;
    public activation: ActivationType;
    public optim: tf.Optimizer;
    public lr: number;

    constructor(observationSpace: spaces.Space, actionSpace: spaces.Space, modelParams: BaseModelParams) {
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;
        this.lr = modelParams.lr;
        this.activation = modelParams.activation;
        if(modelParams.optim !== undefined) {
            this.optim = modelParams.optim;
            // this.optim.getConfig();
            // tf.Optimizer.fromConfig(tf.AdamOptimizer, this.optim.getConfig())

        } else {
            this.optim = tf.train.adam(this.lr);
        }
    }
}

export abstract class BasePolicy extends baseModel {
    abstract predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor;
}

export class ActorCriticPolicy extends BasePolicy {
    public policyModel: tf.LayersModel;
    public valueModel: tf.LayersModel;

    private actionDist: Distribution;
    private logStdInit: number;

    constructor(observationSpace: spaces.Space, actionSpace: spaces.Space, modelParams: ActorCriticParams) {
        super(observationSpace, actionSpace, modelParams);
        this.lr = modelParams.lr;
        this.logStdInit = modelParams.logStdInit;
        if(actionSpace instanceof spaces.Discrete) {
            this.actionDist = new CategoricalDistribution(actionSpace.n);
        } else if(this.actionSpace instanceof spaces.Box) {
            const size = actionSpace.shape.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
            this.actionDist = new DiagGaussianDistribution(size, this.logStdInit);
            (this.actionDist as DiagGaussianDistribution).makeLogStd();
        } else {
            throw new Error("Not supported!");
        }

        this.policyModel = this.createActor(modelParams.policyNet);
        this.valueModel = this.createCritic(modelParams.valueNet);
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
            // if(this.actionDist instanceof CategoricalDistribution) {
            //     this.actionDist.probaDistribution(actionLogits);
            // } else if(this.actionDist instanceof DiagGaussianDistribution) {
            //     this.actionDist.probaDistribution(actionLogits);
            // }
            
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
            actions = actions.reshape(this.actionSpace.shape);

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
        } else if(this.actionSpace instanceof spaces.Box) {
            const size = this.actionSpace.shape.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
            l = tf.layers.dense({
                units: size 
            }).apply(l) as tf.SymbolicTensor;
        } else {
            throw new Error('Unknown action space');
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

