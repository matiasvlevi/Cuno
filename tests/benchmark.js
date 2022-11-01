const fs = require('node:fs');
const { warmup } = require('./utils/warmup.js');
const { Matrix } = require('../Dann/build/dann.js');
const Cuno = require('../build/Release/cuno');

let data = {};

// Warmup GPU Device
warmup(Cuno.dot);

// Iterate from 2 to 1024 and multiply by 2 every iteration 
for (let N = 2; N <= (1 << 10);  N = N << 1) {
  
  // GPU Device function
  let cuno_startTime = new Date().getTime();

  for (let i = 0; i < 1; i++) {
      console.log(Cuno.dot(
        new Array(N).fill(new Array(N).fill(1)),
        new Array(N).fill(new Array(N).fill(1))
      )[0]);
  }

  let cuno_time = new Date().getTime() - cuno_startTime;
  
  // JS CPU function

  let a = new Matrix(N, N);
  a.initiate(1);

  let b = new Matrix(N, N);
  b.initiate(1);

  let js_startTime = new Date().getTime();
  for (let i = 0; i < 1; i++) {
      console.log(Matrix.mult(
        a, b
      ).matrix[0]);
  }
  let js_time = new Date().getTime() - js_startTime;

  // Report 
  console.log(`Matrix dimension (n * n): ${N}`);
  console.log(`--- CUNO: ${cuno_time}ms`);
  console.log(`--- JS  : ${js_time}ms`);

  // Save dava 
  data[`${N}`] = { cuno_time, js_time, N };

};

// Save csv file as benchark.csv
let csv = 'Iterations, CUNO, JS \n';
for (let key in data) {
  csv += `${key},${data[key].cuno_time},${data[key].js_time}\n`; 
}
fs.writeFileSync('benchmark.csv', csv, 'utf-8');

