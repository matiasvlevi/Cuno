const { Dann } = require('../Dann/build/dann.js');
const Cuno = require('../build/Release/cuno');

const model = new Dann(12, 2);

model.addHiddenLayer(8);
model.addHiddenLayer(4);
model.makeWeights();

Cuno.train(model);
