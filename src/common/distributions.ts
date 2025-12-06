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
            actions = actions.reshape([-1]);
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

export class DiagGaussianDistribution extends Distribution {
    private actionDim: number;
    private meanActions: tf.Tensor | null;
    private logStd: tf.Variable | null;
    private logStdInit: number;

    constructor(actionDim: number, logStdInit: number) {
        super();
        this.actionDim = actionDim;
        this.meanActions = null;
        this.logStdInit = logStdInit;
        this.logStd = null;
    }

    makeLogStd() {
        if(this.logStd !== null) {
            tf.dispose(this.logStd);
        }
        this.logStd = tf.variable(tf.fill([this.actionDim], this.logStdInit), true, 'logStd');
    }

    probaDistribution(meanActions: tf.Tensor) {
        // this.meanActions = meanActions;
        // this.logStd = logStd;
        this.meanActions = meanActions;
    }

    // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
    logProb(actions: tf.Tensor): tf.Tensor {
        return tf.tidy(() => {
            if(this.logStd === null || this.meanActions === null || this.logStd === null) {
                throw Error("Values logStd, meanActions and logStd must not be null.");
            }

            this.logStd.print()

            const logUnnormalized = tf.mul(
                -0.5,
                tf.square(
                    tf.sub(
                        tf.div(actions, tf.exp(this.logStd)),
                        tf.div(this.meanActions, tf.exp(this.logStd))
                    )
                )
            )
            const logNormalization = tf.add(
                tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
                tf.log(tf.exp(this.logStd))
            )
            return tf.sum(
                tf.sub(logUnnormalized, logNormalization),
                logUnnormalized.shape.length - 1
            )
        })
    }

/*
    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
*/

    entropy(): tf.Tensor {
        return tf.tidy(() => {
            if(this.meanActions === null || this.logStd === null) {
                throw new Error("Something went wrong!");
            }

            const entropy = tf.add(
                tf.log(this.logStd),
                tf.scalar(0.5 + 0.5 * Math.log(2 * Math.PI))
            )

            return sumIndependentDims(entropy);
        })
    }

    sample(): tf.Tensor {
        return tf.tidy(() => {
            if(this.meanActions === null || this.logStd === null) {
                throw new Error("Something went wrong!");
            }

            const action = tf.add(
                tf.mul(
                    tf.randomNormal([this.actionDim], 0, 1), 
                    tf.exp(this.logStd)
                ),
                this.meanActions
            );
            return action;
        })
    }

    mode(): tf.Tensor {
        if(this.meanActions === null) {
            throw new Error("Something went wrong!");
        }
        return this.meanActions;
    }
}

function sumIndependentDims(tensor: tf.Tensor): tf.Tensor {
    if(tensor.shape.length > 1) {
        return tensor.sum(1);
    } else {
        return tensor.sum();
    }
}