import * as tf from '@tensorflow/tfjs';

// DiagGaussianDistribution
// CategoricalDistribution

export abstract class Distribution {
    abstract probaDistribution(...args: tf.Tensor[]): void;
    abstract logProb(actions: tf.Tensor): tf.Tensor;
    abstract entropy(): tf.Tensor;
    abstract sample(): tf.Tensor;
    abstract mode(): tf.Tensor;

    getActions(deterministic: boolean) {
        if(deterministic) {
            return this.mode();
        } else {
            return this.sample();
        }
    }
}

export class CategoricalDistribution extends Distribution {
    // private actionDim: number;
    private actionLogits: tf.Tensor | null;

    constructor(actionDim: number) {
        super();
        // this.actionDim = actionDim;
        this.actionLogits = null;
    }

    probaDistribution(actionLogits: tf.Tensor) {
        this.actionLogits = actionLogits;
    }

    logProb(actions: tf.Tensor) {
        // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
        return tf.tidy(() => {
            if(this.actionLogits === null) {
                throw new Error("Something went wrong!");
            }
            const numActions = this.actionLogits.shape[this.actionLogits.shape.length - 1]
            const logprobabilitiesAll = tf.logSoftmax(this.actionLogits)
            return tf.sum(
                tf.mul(tf.oneHot(actions, numActions), logprobabilitiesAll),
                logprobabilitiesAll.shape.length - 1
            )
        })
    }

    entropy(): tf.Tensor {
        return tf.tidy(() => {
            if(this.actionLogits === null) {
                throw new Error("Something went wrong!");
            }
            const logprobabilitiesAll = tf.logSoftmax(this.actionLogits.add(tf.scalar(1e-10)));
            return logprobabilitiesAll.sum(-1);
        })
    }

    sample(): tf.Tensor {
        return tf.tidy(() => {
            if(this.actionLogits === null) {
                throw new Error("Something went wrong!");
            }
            let preds = this.actionLogits;
            // if(preds instanceof tf.tensor1d) {
            //     preds = preds as tf.Tensor1D;
            // } else if(preds instanceof tf.tensor2d) {
            //     preds = preds as tf.Tensor2D;
            // } else {
            //     throw new Error("Something went wrong!");
            // }
            let action = tf.multinomial(preds as tf.Tensor2D, 1);
            action = action.reshape([1]);
            return action
        })
    }

    mode(): tf.Tensor {
        if(this.actionLogits === null) {
            throw new Error("Something went wrong!");
        }
        return tf.argMax(this.actionLogits, 1);
    }
}

// class DiagGaussianDistribution extends Distribution {
//     private actionDim: number;
//     private meanActions: tf.Tensor | null;
//     private logStd: tf.Tensor | null;

//     constructor(actionDim: number) {
//         super();
//         this.actionDim = actionDim;
//         this.meanActions = null;
//         this.logStd = null;
//     }

//     probaDistribution(meanActions: tf.Tensor, logStd: tf.Tensor) {
//         this.meanActions = meanActions;
//         this.logStd = logStd;
//     }

//     logProb(actions: tf.Tensor): tf.Tensor {

//     }
// }