const Cuno = require('../build/Release/cuno');


const a = [
  1, 5, 2, 4,
  4, 7, 2, 1,
  8, 4, 5, 2
];

const b = [
  2, 5, 9,
  9, 7, 1,
  1, 4, 5,
  8, 5, 6
];

const c = Cuno.dot(
  a, b, 3, 4, 3
);


console.log(c);
