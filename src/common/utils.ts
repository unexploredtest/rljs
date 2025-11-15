import * as tf from '@tensorflow/tfjs';

// https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
export function clipGradNorm(gradients: tf.NamedTensorMap, maxGradNorm: number): tf.NamedTensorMap {
    return tf.tidy(() => {
        const normedGrad: tf.NamedTensorMap = {};
        for (const key in gradients) {
            const tensorGrad = gradients[key];
            const totalNorm = tf.norm(tensorGrad);
            const clipCoef = tf.scalar(maxGradNorm).div(totalNorm.add(tf.scalar(1e-6)))
            const clipCoefClamped = tf.clipByValue(clipCoef, 0, 1);
            normedGrad[key] = tensorGrad.mul(clipCoefClamped);
        }
        return normedGrad;
    })
}