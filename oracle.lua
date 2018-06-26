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
require 'hdf5'
require 'xlua'
utils = require 'misc.utils'
opt = require 'misc.opts'
print(opt)

if not opt.random then 
  torch.manualSeed(123)
end

local dataset = {}
local h5_file = hdf5.open(opt.data_qas, 'r')
for i, key in pairs({'train', 'valid', 'test'}) do
  dataset['ques_'..key] = h5_file:read('/ques_'..key):all():float()
  dataset['answers_'..key] = h5_file:read('/answers_'..key):all():float()
  dataset['spatial_'..key] = h5_file:read('/spatial_'..key):all():float()
  dataset['category_'..key] = h5_file:read('/category_'..key):all():float()
  _G[key..'Setsize'] = dataset['ques_'..key]:size(1)
  _G['shuffle'..key] = torch.randperm(_G[key..'Setsize'])
end
h5_file:close()

quesLength = dataset['ques_train']:size(2)
spatialLength = dataset['spatial_train']:size(2)
category_size = 90
classes = {'No', 'Yes', 'N/A'}
jsonfile = utils.readJSON(opt.inputJson);
vocabulary_size_ques = utils.count_keys(jsonfile['itow'])
modelPath = opt.modelPath.."Oracle_"..opt.prefix..".t7"

model_ques = nn.Sequential()
model_ques:add(nn.Transpose({1,2}))
lookup = nn.LookupTableMaskZero(vocabulary_size_ques, opt.lookupDim)
model_ques:add(lookup)
if opt.dropout ~= 0 then model_ques:add(nn.Dropout(opt.dropout)) end
lstm = nn.SeqLSTM(opt.lookupDim, opt.hiddenSize)
lstm.maskzero = true
model_ques:add(lstm)
if opt.dropout ~= 0 then model_ques:add(nn.Dropout(opt.dropout)) end
model_ques:add(nn.Select(1, -1))
model_spatial = nn.Sequential()
model_spatial:add(nn.Linear(spatialLength, opt.hiddenSize))
model_spatial:add(nn.ReLU())
model_category = nn.Sequential()
model_category:add(nn.LookupTableMaskZero(category_size, opt.lookupDim))
model_category:add(nn.ReLU())
model_category:add(nn.Linear(opt.lookupDim, opt.hiddenSize))
model_category:add(nn.ReLU())
model_paral = nn.ParallelTable()
model_paral:add(model_ques)
model_paral:add(model_spatial)
model_paral:add(model_category)
model = nn.Sequential()
model:add(model_paral)
model:add(nn.JoinTable(2))
model:add(nn.Linear((3)*opt.hiddenSize, (3)*opt.hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear((3)*opt.hiddenSize, #classes))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

local conTargets, conOutputs
best_valid_accu = 0
best_valid_model = nn.Sequential()
best_train_accu = 0
earlyStopCount = 0

if opt.useCuda then
   require "cutorch"
   require "cunn"
   print("Using GPU:"..opt.deviceId)
   cutorch.setDevice(opt.deviceId)
   model:cuda()
   criterion:cuda()
   best_valid_model:cuda()
else
   print("Not using GPU")
end
parameters, gradParameters = model:getParameters()

if opt.loadModel then
   print("Loading pretrained model: "..modelPath)
   model_load = torch.load(modelPath)
   model_load = model_load:float()
   parameters:copy(model_load:getParameters())
   best_valid_model = model
end

confusion = optim.ConfusionMatrix(classes)
optimState = { learningRate = opt.learningRate,
               learningRateDecay = opt.learningRateDecay }

function load( key, t )
  local k = 1
  local Inputs_ques = torch.FloatTensor(opt.batchSize, quesLength):fill(0)
  local Inputs_spatial = torch.FloatTensor(opt.batchSize, spatialLength):fill(0)
  local Inputs_category = torch.FloatTensor(opt.batchSize):fill(0)
  local Targets = torch.Tensor(opt.batchSize):fill(0)
  for i = t, math.min(t + opt.batchSize -1, _G[key..'Setsize']) do
    Inputs_ques[k] = dataset['ques_'..key][_G['shuffle'..key][i]]
    Inputs_spatial[k] = dataset['spatial_'..key][_G['shuffle'..key][i]]
    Inputs_category[k] = dataset['category_'..key][_G['shuffle'..key][i]]
    Targets[k] = dataset['answers_'..key][_G['shuffle'..key][i]]
    k = k + 1
  end   
  if (t + opt.batchSize -1) > _G[key..'Setsize'] then
    Inputs_ques = Inputs_ques[{{1, (_G[key..'Setsize']+1-t)}}]
    Inputs_spatial = Inputs_spatial[{{1, (_G[key..'Setsize']+1-t)}}]
    Inputs_category = Inputs_category[{{1, (_G[key..'Setsize']+1-t)}}]
    Targets = Targets[{{1, _G[key..'Setsize']+1-t}}]
  end
  if opt.useCuda then
    Inputs_ques = Inputs_ques:cuda()
    Inputs_spatial = Inputs_spatial:cuda()
    Inputs_category = Inputs_category:cuda()
    Targets = Targets:cuda()
  end       
  Inputs = {Inputs_ques, Inputs_spatial, Inputs_category}
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
    step = step + 1
  end
  evaluation( 'valid' )
  print("e:", epoch, "s:", step, "Best valid accuracy: ".. string.format("%.4f", best_valid_accu) ..
    " current accu: ".. string.format("%.4f", confusion.totalValid))
  if best_valid_accu <= confusion.totalValid then
    best_valid_accu = confusion.totalValid
    earlyStopCount = 0
    best_valid_model = model:clone()
    if opt.saveModel then
      best_valid_model = best_valid_model:float()
      torch.save(modelPath, best_valid_model)
    end
    evaluation( 'test' )
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
print("Best validation model TestSet confusion:")
print(confusion)