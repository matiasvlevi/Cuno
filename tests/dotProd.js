const Cuno = require('../build/Release/cuno');

const a = [
  [1, 5, 2, 4],
  [4, 7, 2, 1],
  [8, 4, 5, 2],
  [1, 1, 2, 3]
];

const b = [
  [1],
  [1],
  [1],
  [1]
];

let output = Cuno.matVecDot(
  a, b
);

console.log(output)

output = Cuno.matVecDot(
  output, [[3]]
);

console.log(output)
output = Cuno.matVecDot(
  a, output
);

console.log(output)

output = Cuno.matVecDot(
  output, [[0]]
);

console.log(output)

