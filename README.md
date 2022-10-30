# Cuno | Dannjs Cuda Addon

A node addon for [Dannjs](https://dannjs.org). 

Provides cuda bindings, kernel maps and device memory managment for [Dannjs](https://dannjs.org) computations.

The goal is to speed up `Dann.prototype.backpropagate` by implementing a batch system (Instead of training case by case, we would train a whole dataset batch at once). The current alias `Dann.prototype.train` would also take in a different set of arguments, for batch/gpu training  

A Cuda kernel map would compute all model changes throughout the batch. This is to reduce Memcpy's in the device's memory, thus help reduce training times along with the cuda parallelisation.

<br/>

### Install

```
git clone https://github.com/matiasvlevi/Cuno.git
cd Cuno

npm run init
```

<br/>

### Build

The build configuration may not be supported on your system, please submit `binding.gyp` changes to allow for a broader range of systems 

Build CUDA source with node-gyp (nvcc)

```
npm run build
```

Build the Dannjs Source

```
cd Dann
npm run build:fix
```

(optional) run Dannjs unit tests

```
cd Dann
npm run test
```

<br/>


### Run/Test

Run Javascript tests

`benchmark` will create a `benchmark.csv` file containing performance results.

```
npm run benchmark
npm run test
```

<br/>

### Performance

Here is a logarithmic graph comparing matrix dot products with the Cuno Addon and with native JS 

![Image](https://i.ibb.co/gPfKKHn/Cuno-Log-Graph.png)


<br/>

---

# Current APIs

These are the current stable bindings, not the final target bindings 

## Nodejs API

```js
const Cuno = require('cuno');

const a = [
  [1, 3, 1],
  [2, 4, 6],
  [4, 1, 2],
  [3, 2, 4]
];

const b = [
  [3, 2, 1, 3],
  [5, 1, 1, 4],
  [4, 9, 1, 2]
];

let c = Cuno.dot(a, b);
console.log(c);
```

## CPP API

Allocate & Initialize a neural network

```cpp
const LENGTH = 5
const int ARCH[LENGTH] = { 
  32 * 32 * 3,
  32 * 32,
  24 * 24,
  16 * 16,
  10
};

// Allocate Model
Cuno::DeviceDann<double> *nn = new DeviceDann(ARCH, LENGTH);

// Memory transfer
nn->toDevice(
  // * weights, biases, layers, errors, gradients  * //
);

// Feed Forward 
double inputs[ARCH[0]] = {};
Cuno::Wrappers::ffw(nn, inputs);




```