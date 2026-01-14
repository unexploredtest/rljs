import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';

export abstract class Distribution {
  abstract probaDistribution(...args: tfCore.Tensor[]): void;
  abstract logProb(actions: tfCore.Tensor): tfCore.Tensor;
  abstract entropy(): tfCore.Tensor;
  abstract sample(): tfCore.Tensor;
  abstract mode(): tfCore.Tensor;

  getActions(deterministic: boolean) {
    if (deterministic) {
      return this.mode();
    } else {
      return this.sample();
    }
  }
}

export class CategoricalDistribution extends Distribution {
  private actionLogits: tfCore.Tensor | null;

  constructor() {
    super();
    this.actionLogits = null;
  }

  probaDistribution(actionLogits: tfCore.Tensor) {
    this.actionLogits = actionLogits;
  }

  logProb(actions: tfCore.Tensor) {
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

  entropy(): tfCore.Tensor {
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

  sample(): tfCore.Tensor {
    return tf.tidy(() => {
      if (this.actionLogits === null) {
        throw new Error('Something went wrong!');
      }
      let preds = this.actionLogits;
      let action = tf.multinomial(preds as tfCore.Tensor2D, 1);
      action = action.reshape([1]);
      return action;
    });
  }

  mode(): tfCore.Tensor {
    if (this.actionLogits === null) {
      throw new Error('Something went wrong!');
    }
    return tf.argMax(this.actionLogits, 1);
  }
}

export class MultiCategoricalDistribution extends Distribution {
  private actionDim: number[];
  // private actionLogits: tfCore.Tensor | null;
  private catDists: CategoricalDistribution[];

  constructor(actionDim: number[]) {
    super();
    this.actionDim = actionDim;
    this.catDists = [];
  }

  probaDistribution(actionLogits: tfCore.Tensor) {
    this.catDists = tf.split(actionLogits, this.actionDim, 1).map((split) => {
      const catDist = new CategoricalDistribution();
      catDist.probaDistribution(split);
      return catDist;
    });
  }

  logProb(actions: tfCore.Tensor) {
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

  entropy(): tfCore.Tensor {
    return tf.tidy(() => {
      return tf
        .stack(
          this.catDists.map((catDis) => catDis.entropy()),
          1
        )
        .sum(1);
    });
  }

  sample(): tfCore.Tensor {
    return tf.tidy(() => {
      return tf.stack(
        this.catDists.map((catDis) => catDis.sample()),
        1
      );
    });
  }

  mode(): tfCore.Tensor {
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
  private meanActions: tfCore.Tensor | null;
  public logStd: tfCore.Variable | null;
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

  probaDistribution(meanActions: tfCore.Tensor) {
    // this.meanActions = meanActions;
    // this.logStd = logStd;
    this.meanActions = meanActions;
  }

  // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
  logProb(actions: tfCore.Tensor): tfCore.Tensor {
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

  entropy(): tfCore.Tensor {
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

  sample(): tfCore.Tensor {
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

  mode(): tfCore.Tensor {
    if (this.meanActions === null) {
      throw new Error('Something went wrong!');
    }
    return this.meanActions;
  }
}

function sumIndependentDims(tensor: tfCore.Tensor): tfCore.Tensor {
  if (tensor.shape.length > 1) {
    return tensor.sum(1);
  } else {
    return tensor.sum();
  }
}
