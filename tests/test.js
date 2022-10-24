const { Dann } = require('../Dann/build/dann.js');
const Cuno = require('../build/Release/cuno');

const model = new Dann(8, 2);

model.addHiddenLayer(6);
model.addHiddenLayer(4);
model.makeWeights();

Cuno.dot(model);
