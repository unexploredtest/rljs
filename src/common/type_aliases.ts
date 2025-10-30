import * as tf from '@tensorflow/tfjs';

export type RolloutBufferSamples = [
    observations: tf.Tensor,
    actions: tf.Tensor,
    oldValues: tf.Tensor,
    oldLogProb: tf.Tensor,
    advantages: tf.Tensor,
    returns: tf.Tensor
]