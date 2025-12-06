import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import { PendulumEnv } from "gymjs/envs/classic_control";
import { PendulumEnv as lolerer } from "gymjs/envs/classic_control";
import { TimeLimit } from 'gymjs/wrappers';
import { PPO } from "./algos/ppo.js";


async function run() {
    await tf.setBackend('wasm');
    await tf.ready();
    // await tf.setBackend('webgpu');
    let lol1 = new PendulumEnv();
    let lol = new TimeLimit(lol1, 200);

    let algo = new PPO(lol, {}, {});
    algo.learn(1000_000)
    .then(() => run());

    async function run() {

        // let loler = new lolerer(false, "human");

        // let reward = 0;
        // let done = false;

        // let [obs, info] = loler.reset();
        while(true) {
            let xd1 = new lolerer(10, "human");
            let xd = new TimeLimit(xd1, 200);
            let [obs2, _] = xd.reset();
            let obs = tf.tensor(obs2.dataSync());
            let doner = false;
            let i = 0;
            while(!doner) {
                i++;
                let lool = algo.predict(obs, true);
                let actioner = lool.asScalar().dataSync();
                let action = actioner[0];
                let [obser, reward, done, trun, ___] = await xd.step(lool);
                doner = done || trun;
                obs = tf.tensor(obser.dataSync());
                console.log(reward)
            }
            console.log("final: " + i);
            xd.close();
            // Select all canvas elements
            // const canvases = document.querySelectorAll('canvas');

            // // Remove each canvas element
            // canvases.forEach(canvas => {
            //     canvas.remove();
            // });
        }
    }
}

run();


// Set the backend to webgpu and wait for the module to be ready.
// tf.setBackend('webgpu').then(() => {

// });


// run()
//     .then(() => console.log("finish"));


    // https://stackoverflow.com/questions/79313934/typeerror-0-util-1-isnullorundefined-is-not-a-function-using-tensorflow-tfpn
// console.log("lol")
