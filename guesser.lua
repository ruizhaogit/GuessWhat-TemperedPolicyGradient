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

require 'optim'
require 'rnn'
require 'nngraph'
require 'hdf5'
require 'xlua'
require 'misc.MaskSoftMax'
local cjson = require 'cjson' 
utils = require 'misc.utils'
opt = require 'misc.opts'
print(opt)

if not opt.random then 
  torch.manualSeed(123)
end

jsonfile = utils.readJSON(opt.inputJson);
vocabulary_size_ques = utils.count_keys(jsonfile['itow']) -2 -- leave out START and END token

itow = jsonfile["itow"]
wtoi = jsonfile["wtoi"]
assert( itow[tostring(1)] == "writings" )
assert( wtoi["<START>"] == vocabulary_size_ques + 1 )
answers = {'no', 'yes', 'n/a'}

local dataset = {}
local h5_file = hdf5.open(opt.data_qas, 'r')
for i, key in pairs({'train', 'valid', 'test'}) do
  dataset['dial_rounds_'..key] = h5_file:read('/dial_rounds_'..key):all():float()
  dataset['categoryList_'..key] = h5_file:read('/categoryList_'..key):all():float()
  dataset['spatialList_'..key] = h5_file:read('/spatialList_'..key):all():float()
  dataset['objectLabel_'..key] = h5_file:read('/objectLabel_'..key):all():float()
  _G[key..'Setsize'] = dataset['objectLabel_'..key]:size(1)
  _G['shuffle'..key] = torch.randperm(_G[key..'Setsize'])
end
h5_file:close()

if opt.debug then
  trainSetsize = 20
  validSetsize = 20
  testSetsize = 20
  opt.batchSize = 3
  opt.epochs = 2
  opt.hiddenSize = 4
end

spatialLength = dataset['spatialList_train']:size(3)
numClasses = dataset['spatialList_train']:size(2)
modelPath = opt.modelPath.."Guesser_"..opt.prefix..".t7"

dofile('misc/model_guesser.lua')

output_2 =
  { model_main_2, model_mask }
- nn.CAddTable()
- nn.LogSoftMax()

table.insert(outputs, output_2)
local model = nn.gModule(inputs, outputs)

collectgarbage()

criterion = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(numClasses)

if opt.useCuda then
   require "cutorch"
   require "cunn"
   print("Using GPU:"..opt.deviceId)
   cutorch.setDevice(opt.deviceId)
   model:cuda()
   criterion:cuda()
else
   print("Not using GPU")
end

parameters, gradParameters = model:getParameters()

if opt.loadModel then
   print("Loading pretrained model: "..modelPath)
   parameters:copy( torch.load(modelPath):getParameters() )
end

optimState = { learningRate = opt.learningRate,
               learningRateDecay = opt.learningRateDecay }

local conTargets, conOutputs
best_valid_accu = 0
best_valid_model = model
best_train_accu = 0
earlyStopCount = 0

function load( key, t )
  local k = 1
  local Inputs_dial = torch.FloatTensor(opt.batchSize, 14, 14):fill(0)
  local Inputs_spatialList = torch.FloatTensor(opt.batchSize, 20, spatialLength):fill(0)
  local Inputs_categoryList = torch.FloatTensor(opt.batchSize, 20):fill(0)
  local Targets = torch.Tensor(opt.batchSize):fill(0)
  for i = t, math.min(t + opt.batchSize -1, _G[key..'Setsize']) do
    Inputs_dial[k] = dataset['dial_rounds_'..key][_G['shuffle'..key][i]]
    Inputs_spatialList[k] = dataset['spatialList_'..key][_G['shuffle'..key][i]]
    Inputs_categoryList[k] = dataset['categoryList_'..key][_G['shuffle'..key][i]]
    Targets[k] = dataset['objectLabel_'..key][_G['shuffle'..key][i]]
    k = k + 1
  end 
  if (t + opt.batchSize -1) > _G[key..'Setsize'] then
    Inputs_dial = Inputs_dial[{{1, (_G[key..'Setsize']+1-t)}}]
    Inputs_spatialList = Inputs_spatialList[{{1, (_G[key..'Setsize']+1-t)}}]
    Inputs_categoryList = Inputs_categoryList[{{1, (_G[key..'Setsize']+1-t)}}]
    Targets = Targets[{{1, (_G[key..'Setsize']+1-t)}}]
  end
  maxLen = Inputs_categoryList:nonzero()[{{},{2}}]:max(1)[1][1]
  Inputs_spatialList = Inputs_spatialList[{{},{1, maxLen}}]
  Inputs_categoryList = Inputs_categoryList[{{},{1, maxLen}}]
  if opt.useCuda then
    Inputs_dial = Inputs_dial:cuda()
    Inputs_spatialList = Inputs_spatialList:cuda()
    Inputs_categoryList = Inputs_categoryList:cuda()
    Targets = Targets:cuda()
  end  
  Inputs = {Inputs_dial, Inputs_spatialList, Inputs_categoryList}
  return Inputs, Targets
end

function add2confusion( key, outputs )
  if opt.useCuda then
    conOutputs = outputs:cuda()
    conTargets = _G[key..'Targets']:cuda()
  else
    conOutputs = outputs
    conTargets = _G[key..'Targets']
  end
  confusion:batchAdd(conOutputs, conTargets)
end

function evaluation( key )
  model:evaluate()
  confusion:zero()
  for t = 1, _G[key..'Setsize'], opt.batchSize do
    _G[key..'Inputs'], _G[key..'Targets'] = load(key, t)
    utils.adjustModel( model )
    local outputs = model:forward(_G[key..'Inputs'])
    add2confusion( key, outputs )
  end
  confusion:updateValids()
end

for epoch=1, opt.epochs do
  confusion:zero()
  model:training()
  step = 1
  for t = 1, trainSetsize, opt.batchSize do
    xlua.progress(torch.floor(t/opt.batchSize), torch.floor(trainSetsize/opt.batchSize)) 
    trainInputs, trainTargets = load('train', t)
    utils.adjustModel( model )
    local feval = function()
      gradParameters:zero()
      local outputs = model:forward(trainInputs)
      local f = criterion:forward(outputs, trainTargets)
      local df_do = criterion:backward(outputs, trainTargets)
      model:backward(trainInputs, df_do)
      add2confusion( 'train', outputs )
      return f, gradParameters
    end
    optim.adam(feval, parameters, optimState)
    confusion:updateValids()
    if best_train_accu <= confusion.totalValid then
      best_train_accu = confusion.totalValid
    end
    step = step + 1
  end
  evaluation( 'valid' )
  confusion:updateValids()
  print("e:", epoch, "s:", step, "Best valid accuracy: ".. string.format("%.4f", best_valid_accu) ..
    " current accu: ".. string.format("%.4f", confusion.totalValid))
  if best_valid_accu <= confusion.totalValid then
    best_valid_accu = confusion.totalValid
    earlyStopCount = 0
    best_valid_model = model:clone()
    best_valid_model:clearState()
    best_valid_model = best_valid_model:float()
    if opt.saveModel then
      torch.save(modelPath, best_valid_model)
    end
    evaluation( 'test' )
    confusion:updateValids()
    print("e:", epoch, "s:", step, "Test accuracy: ".. string.format("%.4f", confusion.totalValid))     
  else
    earlyStopCount = earlyStopCount + 1
  end
  if earlyStopCount >= opt.earlyStopThresh then
    print("Early stopping at epoch: " .. tostring(epoch))
    break
  end
end
model = best_valid_model
if opt.useCuda then model = model:cuda() end
evaluation( 'test' )
confusion:updateValids()
print("Final Test accuracy: ".. string.format("%.4f", confusion.totalValid))