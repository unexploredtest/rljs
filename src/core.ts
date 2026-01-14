import * as tfCore from '@tensorflow/tfjs';

let tf = tfCore;

async function changeTensorflow(tensorflow: 'core') {
  if (tensorflow === 'core') {
    tf = tfCore;
  } else {
    throw new Error(`${tensorflow} is unsupported`);
  }
}

export { tf, tfCore, changeTensorflow };
