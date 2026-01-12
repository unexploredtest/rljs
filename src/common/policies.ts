import * as tf from '@tensorflow/tfjs';
import * as gym from '@unexploredtest/gymjs';

import {
  Distribution,
  CategoricalDistribution,
  DiagGaussianDistribution,
  MultiCategoricalDistribution,
} from './distributions';
import {
  BaseModelParams,
  ActorCriticParams,
  ActivationType,
  QNetworkParams,
} from './type_aliases';

export abstract class baseModel<
  ObsType extends tf.Tensor,
  ActType extends tf.Tensor | number,
> {
  protected observationSpace: gym.spaces.Space<ObsType>;
  protected actionSpace: gym.spaces.Space<ActType>;
  public activation: ActivationType;
  public optim: tf.Optimizer;
  public lr: number;

  constructor(
    observationSpace: gym.spaces.Space<ObsType>,
    actionSpace: gym.spaces.Space<ActType>,
    modelParams: BaseModelParams
  ) {
    this.observationSpace = observationSpace;
    this.actionSpace = actionSpace;
    this.lr = modelParams.lr;
    this.activation = modelParams.activation;
    if (modelParams.optim !== undefined) {
      this.optim = modelParams.optim;
      // this.optim.getConfig();
      // tf.Optimizer.fromConfig(tf.AdamOptimizer, this.optim.getConfig())
    } else {
      this.optim = tf.train.adam(this.lr);
    }
  }

  // https://stackoverflow.com/questions/52355693/tensorflow-js-save-model-using-node
  abstract saveModels(...URLs: string[]): Promise<void>;
  abstract loadModels(...URLs: string[]): Promise<void>;
}

export abstract class BasePolicy<
  ObsType extends tf.Tensor,
  ActType extends tf.Tensor | number,
> extends baseModel<ObsType, ActType> {
  abstract predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor;
}

export class ActorCriticPolicy<
  ObsType extends tf.Tensor,
  ActType extends tf.Tensor | number,
> extends BasePolicy<ObsType, ActType> {
  public policyModel: tf.LayersModel;
  public valueModel: tf.LayersModel;

  private actionDist: Distribution;
  private logStdInit: number;

  constructor(
    observationSpace: gym.spaces.Space<ObsType>,
    actionSpace: gym.spaces.Space<ActType>,
    modelParams: ActorCriticParams
  ) {
    super(observationSpace, actionSpace, modelParams);
    this.lr = modelParams.lr;
    this.logStdInit = modelParams.logStdInit;
    if (actionSpace instanceof gym.spaces.Discrete) {
      this.actionDist = new CategoricalDistribution();
    } else if (this.actionSpace instanceof gym.spaces.Box) {
      const size = actionSpace.shape.reduce(
        (accumulator: number, currentValue: number) =>
          accumulator * currentValue,
        1
      );
      this.actionDist = new DiagGaussianDistribution(size, this.logStdInit);
      (this.actionDist as DiagGaussianDistribution).makeLogStd();
    } else if (this.actionSpace instanceof gym.spaces.MultiDiscrete) {
      this.actionDist = new MultiCategoricalDistribution(
        Array.from(this.actionSpace.nVec.reshape([-1]).dataSync())
      );
    } else if (this.actionSpace instanceof gym.spaces.MultiBinary) {
      // We treat multibinary the same as multidiscrete
      const nVec = tf.ones(this.actionSpace.shape).mul(2);
      this.actionDist = new MultiCategoricalDistribution(
        Array.from(nVec.reshape([-1]).dataSync())
      );
    } else {
      throw new Error('Not supported!');
    }

    this.policyModel = this.createActor(modelParams.policyNet);
    this.valueModel = this.createCritic(modelParams.valueNet);
  }

  forward(
    obs: tf.Tensor,
    deterministic: boolean
  ): [tf.Tensor, tf.Tensor, tf.Tensor] {
    return tf.tidy(() => {
      if (obs.shape.length === 1) {
        obs = obs.reshape([-1, ...obs.shape]);
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
    });
  }

  predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
    return tf.tidy(() => {
      if (obs.shape.length === 1) {
        obs = obs.reshape([-1, ...obs.shape]);
      }
      let actionLogits = this.policyModel.predict(obs) as tf.Tensor;
      this.actionDist.probaDistribution(actionLogits);
      let actions = this.actionDist.getActions(deterministic);
      actions = actions.reshape(this.actionSpace.shape);

      return actions;
    });
  }

  predictValues(obs: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      if (obs.shape.length === 1) {
        obs = obs.reshape([-1, ...obs.shape]);
      }
      let values = this.valueModel.predict(obs) as tf.Tensor;
      return values;
    });
  }

  evaluateActions(obs: tf.Tensor, actions: tf.Tensor) {
    return tf.tidy(() => {
      if (obs.shape.length === 1) {
        obs = obs.reshape([-1, ...obs.shape]);
      }
      let values = this.valueModel.predict(obs) as tf.Tensor;
      let actionLogits = this.policyModel.predict(obs) as tf.Tensor;
      this.actionDist.probaDistribution(actionLogits);
      let logProb = this.actionDist.logProb(actions);
      let entropy = this.actionDist.entropy();

      return [values, logProb, entropy];
    });
  }

  async saveModels(PolicyURL: string, ValueURL: string): Promise<void> {
    const saveResultPolicy = await this.policyModel.save(PolicyURL);
    const saveResultValue = await this.valueModel.save(ValueURL);
  }

  async loadModels(PolicyURL: string, ValueURL: string): Promise<void> {
    this.policyModel = await tf.loadLayersModel(PolicyURL);
    this.valueModel = await tf.loadLayersModel(ValueURL);
  }

  getLogStd(): tf.Tensor | undefined {
    if (this.actionDist instanceof DiagGaussianDistribution) {
      const logStd = this.actionDist.logStd?.clone();
      if (logStd) {
        return logStd;
      }
    }

    return undefined;
  }

  // https://github.com/zemlyansky/ppo-tfjs/blob/main/ppo.js
  private createActor(netArch: number[]) {
    const input = tf.layers.input({ shape: this.observationSpace.shape });
    let l = input;
    netArch.forEach((units, i) => {
      l = tf.layers
        .dense({
          units,
          activation: 'relu',
        })
        .apply(l) as tf.SymbolicTensor;
    });
    if (this.actionSpace instanceof gym.spaces.Discrete) {
      l = tf.layers
        .dense({
          units: this.actionSpace.n,
        })
        .apply(l) as tf.SymbolicTensor;
    } else if (this.actionSpace instanceof gym.spaces.Box) {
      const size = this.actionSpace.shape.reduce(
        (accumulator: number, currentValue: number) =>
          accumulator * currentValue,
        1
      );
      l = tf.layers
        .dense({
          units: size,
        })
        .apply(l) as tf.SymbolicTensor;
    } else if (this.actionSpace instanceof gym.spaces.MultiDiscrete) {
      const size = this.actionSpace.nVec.sum().asScalar().dataSync()[0];
      l = tf.layers
        .dense({
          units: size,
        })
        .apply(l) as tf.SymbolicTensor;
    } else if (this.actionSpace instanceof gym.spaces.MultiBinary) {
      // we treat multibinary the same as multidiscrete
      const nVec = tf.ones(this.actionSpace.shape).mul(2);
      const size = nVec.sum().asScalar().dataSync()[0];
      l = tf.layers
        .dense({
          units: size,
        })
        .apply(l) as tf.SymbolicTensor;
    } else {
      throw new Error('Unknown action space');
    }
    return tf.model({ inputs: input, outputs: l });
  }

  private createCritic(netArch: number[]) {
    const input = tf.layers.input({ shape: this.observationSpace.shape });
    let l = input;
    netArch.forEach((units, i) => {
      l = tf.layers
        .dense({
          units: units,
          activation: 'relu',
        })
        .apply(l) as tf.SymbolicTensor;
    });
    l = tf.layers
      .dense({
        units: 1,
        activation: 'linear',
      })
      .apply(l) as tf.SymbolicTensor;
    return tf.model({ inputs: input, outputs: l });
  }
}

export class QNetworkPolicy<
  ObsType extends tf.Tensor,
  ActType extends tf.Tensor | number,
> extends BasePolicy<ObsType, ActType> {
  public model: tf.LayersModel;

  constructor(
    observationSpace: gym.spaces.Space<ObsType>,
    actionSpace: gym.spaces.Space<ActType>,
    modelParams: QNetworkParams
  ) {
    super(observationSpace, actionSpace, modelParams);

    this.model = this.createModel(modelParams.net);
  }

  forward(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
    return this.model.predict(obs) as tf.Tensor;
  }

  predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
    return tf.tidy(() => {
      const qValues = this.forward(obs, deterministic);
      const action = qValues.argMax(1).reshape([qValues.shape[0], -1]).asType("float32");
      return action;
    });
  }

  async saveModels(modelURL: string): Promise<void> {
    const saveResultPolicy = await this.model.save(modelURL);
  }

  async loadModels(modelURL: string): Promise<void> {
    this.model = await tf.loadLayersModel(modelURL);
  }

  private createModel(netArch: number[]) {
    const input = tf.layers.input({ shape: this.observationSpace.shape });
    let l = input;
    netArch.forEach((units, i) => {
      l = tf.layers
        .dense({
          units,
          activation: 'relu',
        })
        .apply(l) as tf.SymbolicTensor;
    });
    if (this.actionSpace instanceof gym.spaces.Discrete) {
      l = tf.layers
        .dense({
          units: this.actionSpace.n,
        })
        .apply(l) as tf.SymbolicTensor;
    } else {
      throw new Error('Unsupported action space');
    }
    return tf.model({ inputs: input, outputs: l });
  }
}

export class DQNPolicy<
  ObsType extends tf.Tensor,
  ActType extends tf.Tensor | number,
> extends BasePolicy<ObsType, ActType> {
  public qNet: QNetworkPolicy<ObsType, ActType>;
  public qNetTarget: QNetworkPolicy<ObsType, ActType>;

  constructor(
    observationSpace: gym.spaces.Space<ObsType>,
    actionSpace: gym.spaces.Space<ActType>,
    modelParams: QNetworkParams
  ) {
    super(observationSpace, actionSpace, modelParams);

    this.qNet = this.createQNetwork(modelParams);
    this.qNetTarget = this.createQNetwork(modelParams);
  }

  forward(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
    return this.qNet.forward(obs, deterministic) as tf.Tensor;
  }

  predict(obs: tf.Tensor, deterministic: boolean): tf.Tensor {
    return this.qNet.predict(obs, deterministic);
  }

  async saveModels(modelURL: string): Promise<void> {
    await this.qNet.saveModels(modelURL);
  }

  async loadModels(modelURL: string): Promise<void> {
    await this.qNet.loadModels(modelURL);
  }

  private createQNetwork(params: QNetworkParams) {
    return new QNetworkPolicy(this.observationSpace, this.actionSpace, params);
  }
}
