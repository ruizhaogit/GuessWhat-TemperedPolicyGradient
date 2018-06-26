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

require 'rnn'
require 'nngraph'
require 'hdf5'
require 'xlua'
dofile('misc/optim_updates.lua')
local cjson = require 'cjson' 
utils = require 'misc.utils'
opt = require 'misc.opts'
print(opt)

if not opt.random then 
  torch.manualSeed(123)
end

local dataset = {}
local h5_file = hdf5.open(opt.data_qas, 'r')
for i, key in pairs({'train', 'valid', 'test'}) do
  dataset['quesSucc_in_'..key] = h5_file:read('/quesSucc_in_'..key):all():float()
  dataset['quesSucc_out_'..key] = h5_file:read('/quesSucc_out_'..key):all():float()
  dataset['history_'..key] = h5_file:read('/history_'..key):all():float()
  dataset['quesSucc_id_'..key] = h5_file:read('/quesSucc_id_'..key):all():float()
  _G[key..'Setsize'] = dataset['history_'..key]:size(1)
  _G['shuffle'..key] = torch.randperm(_G[key..'Setsize'])
end
h5_file:close()
histLength = dataset['history_train']:size()[2]
quesLength = dataset['quesSucc_in_train']:size()[2]

local h5_file = hdf5.open(opt.inputImg, 'r')
dataset['img_feat'] = h5_file:read('/img_feat'):all():float()
dataset['train_img'] = h5_file:read('/train_img'):all():float()
dataset['valid_img'] = h5_file:read('/valid_img'):all():float()
dataset['test_img'] = h5_file:read('/test_img'):all():float()
h5_file:close()

if opt.debug then
  trainSetsize = 5
  validSetsize = 5
  testSetsize = 5
  opt.batchSize = 3
  opt.epochs = 2
end
jsonfile = utils.readJSON(opt.inputJson)
vocabulary_size = utils.count_keys(jsonfile['itow'])
itow = jsonfile["itow"]
wtoi = jsonfile["wtoi"]
assert( itow[tostring(1)] == "writings" )
assert( wtoi["<START>"] == vocabulary_size - 1 )

modelPath = opt.modelPath.."QGen_"..opt.prefix..".t7"

dofile('misc/model_qgen.lua')
dec:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))
criterion = nn.ClassNLLCriterion()
criterion.ignoreIndex = 0
criterion.sizeAverage = false
criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(criterion, 1))

model = nn.Sequential()
model:add(enc)
model:add(dec)
model = require('misc.weight-init')(model, 'xavier')
encoder = model:get(1)
decoder = model:get(2)

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

modelW, modeldW = model:getParameters()
model:training()
optims = {}
optims.learningRate = opt.learningRate
collectgarbage()

if opt.loadModel then
   if opt.useCuda then
      require 'cutorch'
      require 'cunn'
   end
   print("Loading pretrained model")
   modelW:copy( torch.load(modelPath):getParameters() )
   if not opt.useCuda then
      model = model:float()
   else
      model = model:cuda()
   end
end

best_valid_loss = 100000
best_valid_model = model
best_train_loss = 100000
earlyStopCount = 0
valid_loss = 0

function load( key, t )
  local k = 1
  history = torch.FloatTensor(opt.batchSize, histLength):fill(0)
  quesSucc_in = torch.FloatTensor(opt.batchSize, quesLength):fill(0)
  img = torch.FloatTensor(opt.batchSize, opt.imgFeatureSize):fill(0)
  quesSucc_out = torch.FloatTensor(opt.batchSize, quesLength):fill(0)
  quesSucc_id = torch.FloatTensor(opt.batchSize):fill(0)
  for i = t, math.min(t + opt.batchSize -1, _G[key..'Setsize']) do
    history[k] = dataset['history_'..key][_G['shuffle'..key][i]]
    quesSucc_in[k] = dataset['quesSucc_in_'..key][_G['shuffle'..key][i]]
    img[k] = dataset['img_feat'][dataset[key..'_img'][_G['shuffle'..key][i]]]
    quesSucc_out[k] = dataset['quesSucc_out_'..key][_G['shuffle'..key][i]]
    quesSucc_id[k] = dataset['quesSucc_id_'..key][_G['shuffle'..key][i]]
    k = k + 1
  end 
  if (t + opt.batchSize -1) > _G[key..'Setsize']  then
    history = history[{{1, (_G[key..'Setsize']+1-t)}}]
    quesSucc_in = quesSucc_in[{{1, (_G[key..'Setsize']+1-t)}}]
    img = img[{{1, (_G[key..'Setsize']+1-t)}}]
    quesSucc_out = quesSucc_out[{{1, (_G[key..'Setsize']+1-t)}}]
  end
  if opt.useCuda then
    history = history:cuda()
    quesSucc_in = quesSucc_in:cuda()
    img = img:cuda()
    quesSucc_out = quesSucc_out:cuda()
  end  
  inputs = {}
  table.insert(inputs, history:t())
  table.insert(inputs, img)
  quesSucc_in = quesSucc_in:t()
  quesSucc_out = quesSucc_out:t()
  return inputs, quesSucc_in, quesSucc_out, history, img, quesSucc_id
end

for epoch=1, opt.epochs do
  runningLoss = 0
  model:training()
  step = 1
  for t = 1, trainSetsize, opt.batchSize do
    xlua.progress(torch.floor(t/opt.batchSize), torch.floor(trainSetsize/opt.batchSize)) 
    model:zeroGradParameters()
    inputs, quesSucc_in_train, quesSucc_out_train, history_train, train_img, quesSucc_id_train = load( 'train', t )
    local encOut = encoder:forward(inputs)
    utils.forwardConnect(encoder, decoder, encOut, history_train:size(2))
    local curLoss = 0
    local decOut = decoder:forward(quesSucc_in_train)
    curLoss = criterion:forward(decOut, quesSucc_out_train)
    local gradCriterionOut = criterion:backward(decOut, quesSucc_out_train)
    decoder:backward(quesSucc_in_train, gradCriterionOut)
    local gradDecOut = utils.backwardConnect(encoder, decoder)
    encoder:backward(inputs, gradDecOut)
    local numTokens = torch.sum(quesSucc_out_train:gt(0))
    if runningLoss > 0 then
        runningLoss = 0.95 * runningLoss + 0.05 * curLoss/numTokens
    else
        runningLoss = curLoss/numTokens
    end
    modeldW:clamp(-5.0, 5.0)
    adam(modelW, modeldW, optims)
    step = step + 1
    if opt.verbose then utils.greedySearch( history_train, train_img, quesSucc_id_train, "train" ) end
  end
  print("e:", epoch, "s:", step, "lr:", optims.learningRate, "trainLoss", string.format("%.4f", runningLoss))

  if optims.learningRate > opt.minLRate then
      optims.learningRate = optims.learningRate * opt.learningRateDecay
  end

  for t = 1, validSetsize, opt.batchSize do
    model:evaluate()
    local numTokens = 0
    local curLoss = 0
    inputs, quesSucc_in_valid, quesSucc_out_valid, history_valid, valid_img, quesSucc_id_valid = load( 'valid', t )
    local encOut = encoder:forward(inputs)
    utils.forwardConnect(encoder, decoder, encOut, histLength)
    local decOut = decoder:forward(quesSucc_in_valid)
    curLoss = curLoss + criterion:forward(decOut, quesSucc_out_valid)
    local numTokens = numTokens + torch.sum(quesSucc_out_valid:gt(0)) -- element wise >
    curLoss = curLoss / numTokens
    valid_loss = curLoss
  end
  print(string.format('valid-loss: %f\t Perplexity: %f', string.format("%.4f",valid_loss), string.format("%.4f",math.exp(valid_loss))))
  model:training()
  if best_valid_loss >= valid_loss then
    best_valid_loss = valid_loss
    earlyStopCount = 0
    best_valid_model = model:clone()
    best_valid_model:clearState()
    best_valid_model = best_valid_model:float()
    if opt.saveModel then
      torch.save(modelPath, best_valid_model) 
      print("Saving model...")
    end
  else
     earlyStopCount = earlyStopCount + 1
  end 

  if earlyStopCount >= opt.earlyStopThresh then
     print("Early stopping at epoch: " .. tostring(epoch))
     break
  end
  if opt.verbose then utils.greedySearch( history_valid, valid_img, quesSucc_id_valid, "valid" ) end
end