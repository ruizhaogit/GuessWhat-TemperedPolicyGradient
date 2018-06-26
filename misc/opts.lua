--[[ 
__author__ = "Rui Zhao"
__copyright__ = "Siemens AG, 2018"
__licencse__ = "MIT"
__version__ = "0.1"

MIT License
Copyright (c) 2018 Siemens AG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
  ]]

torch.setdefaulttensortype("torch.FloatTensor")
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--modelPath', 'model/', 'Pre trained model path.')
cmd:option('--data_qas', 'data/data_qas.h5', 'H5 file with QAs and other information')
cmd:option('--inputImg', 'data/data_img.h5', 'HDF5 file with image features')
cmd:option('--inputIndex', 'data/id2index.json', 'JSON file with image index')
cmd:option('--inputJson', 'data/vocab.json', 'JSON file with info and vocab')
cmd:option('--useCuda', true, 'Use GPU for training.')
cmd:option('--deviceId', 1, 'Device Id.')
cmd:option('--prefix', '1', 'prefix')
cmd:option('--loadModel', false, 'Load pretrained model and train further.')
cmd:option('--saveModel', true, 'save model')
cmd:option('--epochs', 80, 'maximum number of epochs to run, 30 for supervised and 80 for reinforce')
cmd:option('--earlyStopThresh', 20, 'Early stopping threshold, 5 for oracle and guesser, 10 for qgen, 20 for reinforce.')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--lookupDim', 300, 'Lookup feature dimensionality.')
cmd:option('--imgFeatureSize', 4096, 'Size of the image feature');
cmd:option('--imgEmbedSize', 300, 'Size of the multimodal embedding')
cmd:option('--hiddenSize', 512, 'Hidden size for LSTM.')
cmd:option('--dropout', 0, 'Dropout on hidden representations.')
cmd:option('--learningRate', 0.001, 'Learning rate. 0.0001 for oracle and guesser, 0.001 for qgen and reinforce.')
cmd:option('--learningRateDecay', 1e-7, 'Learning rate decay. 1e-7 for oracle and guesser, 0.5 for qgen')
cmd:option('--minLRate', 5e-5, 'Minimum learning rate')
cmd:option('--debug', false, 'Activate debug mode')
cmd:option('--beamSize', 5, 'beamSize')
cmd:option('--beamLen', 13, 'beamLen')
cmd:option('--verbose', false, 'Show dialogue samples and so on.')
cmd:option('--temperatures', '{1}', 'temperatures {1, 1.5}')
cmd:option('--round', 8, 'round')
cmd:option('--random', false, 'random')
cmd:option('--startEpoch', 1, 'startEpoch')
cmd:option('--DynamicTPG', false, 'DynamicTPG')
cmd:option('--tempMin', 0.5, 'tempMin')
cmd:option('--tempMax', 2.0, 'tempMax')
cmd:option('--saveLastModel', false, 'save last model')
cmd:option('--trainGuesser', true, 'train guesser')
cmd:option('--bestValid', 0, 'previous best validation running average')



cmd:text()
local opt = cmd:parse(arg or {})
return opt

