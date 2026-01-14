import * as tfCore from '@tensorflow/tfjs';
import { tf } from '../core';

// https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
export function clipGradNorm(
  gradients: tfCore.NamedTensorMap,
  maxGradNorm: number
): tfCore.NamedTensorMap {
  return tf.tidy(() => {
    const normedGrad: tfCore.NamedTensorMap = {};
    for (const key in gradients) {
      const tensorGrad = gradients[key];
      const totalNorm = tf.norm(tensorGrad);
      const clipCoef = tf
        .scalar(maxGradNorm)
        .div(totalNorm.add(tf.scalar(1e-6)));
      const clipCoefClamped = tf.clipByValue(clipCoef, 0, 1);
      normedGrad[key] = tensorGrad.mul(clipCoefClamped);
    }
    return normedGrad;
  });
}

export function clipTensor(
  tensor: tfCore.Tensor,
  low: tfCore.Tensor,
  high: tfCore.Tensor
): tfCore.Tensor {
  return tf.tidy(() => {
    const upperFlow = high.sub(tensor);
    const upperFixed = tensor.add(upperFlow);

    const bottomFlow = low.sub(tensor);
    const bottomFixed = tensor.add(bottomFlow);

    tensor = tf.where(upperFlow.less(0), upperFixed, tensor);
    tensor = tf.where(bottomFlow.less(0).logicalNot(), bottomFixed, tensor);

    return tensor;
  });
}

// From stable baselines
export function explainedVariance(
  yPred: tfCore.Tensor,
  yTrue: tfCore.Tensor
): tfCore.Tensor {
  /*
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    */
  return tf.tidy(() => {
    const varY = tf.moments(yTrue.flatten()).variance;
    const varDif = tf.moments(yTrue.flatten().sub(yPred.flatten())).variance;
    return tf.scalar(1).sub(varDif.div(varY));
  });
}

export function tensorToTf(tensor: tfCore.Tensor): tfCore.Tensor {
  return tf.tensor(tensor.dataSync(), tensor.shape, tensor.dtype);
}

export function tensorToCore(tensor: tfCore.Tensor): tfCore.Tensor {
  return tfCore.tensor(tensor.dataSync(), tensor.shape, tensor.dtype);
}
