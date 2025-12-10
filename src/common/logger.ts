import * as tf from '@tensorflow/tfjs';

// export enum Level {
//   Debug,
//   Info,
//   Warn,
//   Error,
//   Disabled
// }

// type NestedRecord = {
//     [key: string]: number | NestedRecord;
// }

export class Logger {
    // public level: Level;
    private nameToValue: Record<string, number>;
    private nameToCount: Record<string, number>;

    constructor() {
        this.nameToValue = {};
        this.nameToCount = {};
    }

    record(key: string, value: number): void {
        this.nameToValue[key] = value;
    }

    recordMean(key: string, value: number): void {
        // Calculate the new mean
        const oldValue = this.nameToValue[key] || 0;
        const count = this.nameToCount[key] || 0;

        this.nameToValue[key] = (oldValue * count / (count + 1)) + (value / (count + 1));
        this.nameToCount[key] = count + 1; // Increment count
    }

    dump(step: number) {
        this.nameToValue["step"] = step;
        const nameToValue = this.nameToValue;
        this.nameToValue = {};
        this.nameToCount = {};

        return nameToValue;
    }
}