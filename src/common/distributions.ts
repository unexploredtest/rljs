import * as tf from '@tensorflow/tfjs';

export abstract class Distribution {
  abstract probaDistribution(...args: tf.Tensor[]): void;
  abstract logProb(actions: tf.Tensor): tf.Tensor;
  abstract entropy(): tf.Tensor;
  abstract sample(): tf.Tensor;
  abstract mode(): tf.Tensor;

  getActions(deterministic: boolean) {
    if (deterministic) {
      return this.mode();
    } else {
      return this.sample();
    }
  }
}

export class CategoricalDistribution extends Distribution {
  private actionLogits: tf.Tensor | null;

  constructor() {
    super();
    this.actionLogits = null;
  }

  probaDistribution(actionLogits: tf.Tensor) {
    this.actionLogits = actionLogits;
  }

  logProb(actions: tf.Tensor) {
    // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
    return tf.tidy(() => {
      if (this.actionLogits === null) {
        throw new Error('Something went wrong!');
      }
      const numActions =
        this.actionLogits.shape[this.actionLogits.shape.length - 1];
      const logprobabilitiesAll = tf.logSoftmax(this.actionLogits);
      actions = actions.reshape([-1]);
      return tf.sum(
        tf.mul(tf.oneHot(actions, numActions), logprobabilitiesAll),
        logprobabilitiesAll.shape.length - 1
      );
    });
  }

  entropy(): tf.Tensor {
    return tf.tidy(() => {
      if (this.actionLogits === null) {
        throw new Error('Something went wrong!');
      }
      const logprobabilitiesAll = tf.logSoftmax(
        this.actionLogits.add(tf.scalar(1e-10))
      );
      return logprobabilitiesAll.sum(-1);
    });
  }

  sample(): tf.Tensor {
    return tf.tidy(() => {
      if (this.actionLogits === null) {
        throw new Error('Something went wrong!');
      }
      let preds = this.actionLogits;
      let action = tf.multinomial(preds as tf.Tensor2D, 1);
      action = action.reshape([1]);
      return action;
    });
  }

  mode(): tf.Tensor {
    if (this.actionLogits === null) {
      throw new Error('Something went wrong!');
    }
    return tf.argMax(this.actionLogits, 1);
  }
}

export class MultiCategoricalDistribution extends Distribution {
  private actionDim: number[];
  // private actionLogits: tf.Tensor | null;
  private catDists: CategoricalDistribution[];

  constructor(actionDim: number[]) {
    super();
    this.actionDim = actionDim;
    this.catDists = [];
  }

  probaDistribution(actionLogits: tf.Tensor) {
    this.catDists = tf.split(actionLogits, this.actionDim, 1).map((split) => {
      const catDist = new CategoricalDistribution();
      catDist.probaDistribution(split);
      return catDist;
    });
  }

  logProb(actions: tf.Tensor) {
    return tf.tidy(() => {
      const size = (actions.shape as number[])[1];
      return tf
        .stack(
          actions
            .split(size, 1)
            .map((split, index) => this.catDists[index].logProb(split)),
          1
        )
        .sum(1);
    });
  }

  entropy(): tf.Tensor {
    return tf.tidy(() => {
      return tf
        .stack(
          this.catDists.map((catDis) => catDis.entropy()),
          1
        )
        .sum(1);
    });
  }

  sample(): tf.Tensor {
    return tf.tidy(() => {
      return tf.stack(
        this.catDists.map((catDis) => catDis.sample()),
        1
      );
    });
  }

  mode(): tf.Tensor {
    return tf.tidy(() => {
      return tf.stack(
        this.catDists.map((catDis) => catDis.mode()),
        1
      );
    });
  }
}

export class DiagGaussianDistribution extends Distribution {
  private actionDim: number;
  private meanActions: tf.Tensor | null;
  public logStd: tf.Variable | null;
  private logStdInit: number;

  constructor(actionDim: number, logStdInit: number) {
    super();
    this.actionDim = actionDim;
    this.meanActions = null;
    this.logStdInit = logStdInit;
    this.logStd = null;
  }

  makeLogStd() {
    if (this.logStd !== null) {
      tf.dispose(this.logStd);
    }
    this.logStd = tf.variable(
      tf.fill([this.actionDim], this.logStdInit),
      true,
      'logStd'
    );
  }

  probaDistribution(meanActions: tf.Tensor) {
    // this.meanActions = meanActions;
    // this.logStd = logStd;
    this.meanActions = meanActions;
  }

  // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
  logProb(actions: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      if (
        this.logStd === null ||
        this.meanActions === null ||
        this.logStd === null
      ) {
        throw Error('Values logStd, meanActions and logStd must not be null.');
      }

      const variance = tf.exp(this.logStd).square();
      const result = tf.sub(
        tf.div(tf.neg(actions.sub(this.meanActions).square()), variance.mul(2)),
        this.logStd.add(Math.log(Math.sqrt(2.0 * Math.PI)))
      );

      return tf.sum(result, 1);
    });
  }

  /*
    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
*/

  entropy(): tf.Tensor {
    return tf.tidy(() => {
      if (this.meanActions === null || this.logStd === null) {
        throw new Error('Something went wrong!');
      }

      const entropy = tf.add(
        this.logStd,
        tf.scalar(0.5 + 0.5 * Math.log(2 * Math.PI))
      );

      return sumIndependentDims(entropy);
    });
  }

  sample(): tf.Tensor {
    return tf.tidy(() => {
      if (this.meanActions === null || this.logStd === null) {
        throw new Error('Something went wrong!');
      }

      const action = tf.add(
        tf.mul(tf.randomNormal([this.actionDim], 0, 1), tf.exp(this.logStd)),
        this.meanActions
      );
      return action;
    });
  }

  mode(): tf.Tensor {
    if (this.meanActions === null) {
      throw new Error('Something went wrong!');
    }
    return this.meanActions;
  }
}

function sumIndependentDims(tensor: tf.Tensor): tf.Tensor {
  if (tensor.shape.length > 1) {
    return tensor.sum(1);
  } else {
    return tensor.sum();
  }
}
