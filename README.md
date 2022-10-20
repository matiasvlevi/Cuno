# Cuno | Dannjs Cuda Addon

A node addon for [Dannjs](https://dannjs.org). 

Provides cuda bindings, kernel maps and device memory managment for [Dannjs](https://dannjs.org) computations.

The goal is to speed up `Dann.prototype.backpropagate` by implementing a batch system (Instead of training case by case, we would train a whole dataset batch at once). A kernel map would then compute all model changes throughout the batch. This is to reduce Memcpy's in the device's memory, thus help reduce training times along with the cuda parallelisation.

<br/>

### Install

```
git clone --recurse-submodules https://github.com/matiasvlevi/Cuno.git

cd Cuno
npm i

cd Dann
git checkout Cuno
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
