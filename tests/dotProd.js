const Cuno = require('../build/Release/cuno');

const a = [
  [1, 5, 2, 4],
  [4, 7, 2, 1],
  [8, 4, 5, 2],
  [1, 1, 2, 3]
];

const b = [
  [2, 5, 9],
  [9, 7, 1],
  [1, 4, 5],
  [8, 5, 6]
];

let output = Cuno.dot(
  a, b
);

console.log(output)

