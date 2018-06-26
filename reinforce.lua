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

require 'nn'
require 'dpnn'
require 'dp'
require 'rnn'
require 'nngraph'
require 'hdf5'
local cjson = require 'cjson' 
require 'xlua'
require 'optim'
dofile("misc/VarianceReducedReward.lua")
dofile("misc/optim_updates.lua")
dofile("misc/Reinforce_rewardGradient.lua")
dofile("misc/ReinforceCategorical_rewardGradient.lua")
dofile("misc/ReinforceCategoricalTemperature_rewardGradient.lua")
color = dofile("misc/color.lua")
require 'misc.MaskSoftMax'
utils = require 'misc.utils'
opt = require 'misc.opts'
print(opt)

BASELINE = 0
REWARD = 0

if not opt.random then 
  torch.manualSeed(123)
end

if opt.useCuda then
  require 'cunn'
  require 'cutorch'
  print("Using GPU:"..opt.deviceId)
  cutorch.setDevice(opt.deviceId)
end

local dataset = {}
local h5_file = hdf5.open(opt.inputImg, 'r')
dataset['img_feat'] = h5_file:read('/img_feat'):all():float()
h5_file:close() 
id2index = utils.readJSON("data/id2index.json")
assert(id2index[tostring(489209)] == 1, "Error: key ~= index")
jsonfile = utils.readJSON(opt.inputJson)
vocabulary_size = utils.count_keys(jsonfile['itow'])
itow = jsonfile["itow"]
wtoi = jsonfile["wtoi"]
assert( itow[tostring(1)] == "writings" )
assert( wtoi["<START>"] == vocabulary_size - 1 )
answers = {'no', 'yes', 'n/a'}

if opt.DynamicTPG then
  DocumCount = 0
  VocabCount = torch.FloatTensor(vocabulary_size):fill(0)
end

local modelpath_questioner = opt.modelPath.."QGen_"..tostring(opt.prefix)..".t7"
local modelpath_oracle = opt.modelPath.."Oracle_".."1"..".t7"
local modelpath_guesser = opt.modelPath.."Guesser_"..tostring(opt.prefix)..".t7"

temperatures = loadstring(" return " .. opt.temperatures)()
dofile('misc/model_qgen.lua')

dec:add(nn.Sequencer(nn.MaskZero(nn.SoftMax(), 1)))
dec:add(nn.Sequencer(nn.MaskZero(nn.ReinforceCategoricalTemperature_rewardGradient(1), 1)))

model_questioner = nn.Sequential()
model_questioner:add(enc)
model_questioner:add(dec)

model_questioner = require('misc.weight-init')(model_questioner, 'xavier')
encoder = model_questioner:get(1)
decoder = model_questioner:get(2)
decoder.rnnLayers[1]._remember = 'both' -- for both train and eval
model_questionerW, model_questionerdW = model_questioner:getParameters()
model_questionerW:copy( torch.load(modelpath_questioner):getParameters() )

criterion_vrd = nn.VarianceReducedReward(model_questioner)
criterion_vrd = nn.SequencerCriterion(nn.MaskZeroCriterion(criterion_vrd, 1))
optims = {}
optims.learningRate = opt.learningRate
model_oracle = torch.load(modelpath_oracle)
model_questioner:evaluate()
model_oracle:evaluate()

collectgarbage()

vocabulary_size_ques = vocabulary_size - 2

dofile('misc/model_guesser.lua')

output_2 =
  { model_main_2, model_mask }
- nn.CAddTable()
- nn.SoftMax()
- nn.ReinforceCategorical_rewardGradient()
   
table.insert(outputs, output_2)
model_guesser = nn.gModule(inputs, outputs)
model_guesserW, model_guesserdW = model_guesser:getParameters()
model_guesserW:copy( torch.load(modelpath_guesser):getParameters() )
model_guesser:evaluate()
criterion_vrd_guesser = nn.VarianceReducedReward(model_guesser)

collectgarbage()

model_oracle = torch.load(modelpath_oracle)
model_questioner:evaluate()
model_oracle:evaluate()

if (opt.epochs == 0) or (not (opt.startEpoch == 0)) then
  best_valid_model_questioner = model_questioner
  best_valid_model_guesser = model_guesser
end

if opt.debug then
  -- opt.epochs = 80
  opt.batchSize = 64
end

function json2vec( key )
  total_count = 0
  local quesSucc_ids = {}
  local imgs = {}
  local Inputs_spatials = {}
  local Inputs_categorys = {}
  local Inputs_spatialLists = {}
  local Inputs_categoryLists = {}
  local labels = {}
  local Id2url = {}

  for line in io.lines('data/guesswhat.'..key..'.jsonl') do
      total_count = total_count + 1
      json_line = cjson.decode(line)
      image_id = json_line['image']['id']
      table.insert(quesSucc_ids, json_line['id'])
      image_fc8 = dataset['img_feat'][id2index[tostring(image_id)]]
      table.insert(imgs, torch.FloatTensor(image_fc8))
      table.insert(Inputs_spatials, torch.FloatTensor(utils.get_normalized_spatial_info(json_line)))
      table.insert(Inputs_categorys, torch.FloatTensor({utils.get_category(json_line)}))
      table.insert(Inputs_spatialLists, utils.get_normalized_spatial_guesser(json_line))
      table.insert(Inputs_categoryLists, utils.get_objectCategoryList(json_line))
      table.insert(labels, utils.get_objectLabel(json_line))
      explore_id = utils.split(json_line['image']['coco_url'], "/")
      Id2url[tostring(json_line['id'])] = 'http://cocodataset.org/#explore?id='..explore_id[#explore_id]
      if opt.debug then
        if total_count > 1024 then break end
      end
  end
  quesSucc_ids = torch.FloatTensor(quesSucc_ids)
  imgs = torch.cat(imgs, 2):t()
  Inputs_spatials = torch.cat(Inputs_spatials, 2):t()
  Inputs_categorys = torch.cat(Inputs_categorys, 2):t():squeeze()
  Inputs_spatialLists =  torch.cat(Inputs_spatialLists, 3)
  Inputs_spatialLists = torch.swapaxes(Inputs_spatialLists, {3,1,2})
  Inputs_categoryLists = torch.cat(Inputs_categoryLists, 2):t()
  labels = torch.FloatTensor(labels)
  Setsize = total_count
  return quesSucc_ids, imgs, Inputs_spatials, Inputs_categorys, Inputs_spatialLists, Inputs_categoryLists, labels, Setsize, Id2url
end

quesSucc_ids_valid, valid_imgs, validInputs_spatials, validInputs_categorys, validInputs_spatialLists, 
validInputs_categoryLists, validlabels, validSetsize, validId2url = json2vec( 'valid' )

batchSize = opt.batchSize
if not opt.useCuda then
  model_questioner = model_questioner:float()
  model_oracle = model_oracle:float()
  model_guesser = model_guesser:float()
else
  model_questioner = model_questioner:cuda()
  model_oracle = model_oracle:cuda()
  model_guesser = model_guesser:cuda()
  model_questionerW, model_questionerdW = model_questioner:getParameters()
  model_questionerW = model_questionerW:cuda()
  model_questionerdW = model_questionerdW:cuda()
  model_guesserW = model_guesserW:cuda()
  model_guesserdW = model_guesserdW:cuda()
end

function load( key, t )
  k = 1
  historys = {}
  historyRounds = {}
  local history = torch.FloatTensor(opt.batchSize, (opt.beamLen*opt.round)):fill(0)
  local historyMemory = torch.FloatTensor(opt.batchSize, 14, 14):fill(0)
  local img = torch.FloatTensor(opt.batchSize, opt.imgFeatureSize):fill(0)
  local quesSucc_id = torch.FloatTensor(opt.batchSize):fill(0)
  local Inputs_spatial = torch.FloatTensor(opt.batchSize, 8):fill(0) -- spatialLength = 8
  local Inputs_category = torch.FloatTensor(opt.batchSize):fill(0)
  local Inputs_spatialList = torch.FloatTensor(opt.batchSize, 20, 8):fill(0)
  local Inputs_categoryList = torch.FloatTensor(opt.batchSize, 20):fill(0)
  local label = torch.FloatTensor(opt.batchSize):fill(0)
  for i = t, math.min(t + opt.batchSize -1, _G[key..'Setsize']) do
      if (key == 'train') and opt.DynamicTPG then DocumCount = DocumCount + 1 end
      img[k] = _G[key..'_imgs'][i]
      quesSucc_id[k] = _G['quesSucc_ids_'..key][i]
      Inputs_spatial[k] = _G[key..'Inputs_spatials'][i]
      Inputs_category[k] = _G[key..'Inputs_categorys'][i]
      Inputs_spatialList[k] = _G[key..'Inputs_spatialLists'][i]
      Inputs_categoryList[k] = _G[key..'Inputs_categoryLists'][i]
      label[k] = _G[key..'labels'][i]
      k = k + 1
  end
  if (t + opt.batchSize -1) > _G[key..'Setsize'] then
      img = img[{{1, (_G[key..'Setsize']+1-t)}}]
      Inputs_spatial = Inputs_spatial[{{1, (_G[key..'Setsize']+1-t)}}]
      Inputs_category = Inputs_category[{{1, (_G[key..'Setsize']+1-t)}}]
      quesSucc_id = quesSucc_id[{{1, (_G[key..'Setsize']+1-t)}}]
      history = history[{{1, (_G[key..'Setsize']+1-t)}}]
      historyMemory = historyMemory[{{1, (_G[key..'Setsize']+1-t)}}]
      Inputs_spatialList = Inputs_spatialList[{{1, (_G[key..'Setsize']+1-t)}}]
      Inputs_categoryList = Inputs_categoryList[{{1, (_G[key..'Setsize']+1-t)}}]
      label = label[{{1, _G[key..'Setsize']+1-t}}]
      opt.batchSize = _G[key..'Setsize']+1-t
  end
  return history, historyMemory, img, quesSucc_id, Inputs_spatial, Inputs_category, Inputs_spatialList, Inputs_categoryList, label
end

function QuesAns( key, stochastic )
  while ((round <= opt.round) and (end_count<opt.batchSize)) do
    Questions = utils.sampling( _G['history_'..key], _G[key..'_img'], _G['quesSucc_id_'..key], key, _G[key..'Id2url'], stochastic)
    if (key == 'train') then model_questioner:training() end
    _G[key..'Inputs_ques'] = {}
    _G[key..'Inputs_ques_in'] = {}
    _G[key..'Inputs_ques_out'] = {}
    for i, q in ipairs(Questions) do 
      table.insert(_G[key..'Inputs_ques_in'], utils.padQuestionRight(torch.FloatTensor(utils.quesToIn(q['quesWords']))))
      table.insert(_G[key..'Inputs_ques_out'], utils.padQuestionRight(torch.FloatTensor(q['quesWords'])))
      table.insert(_G[key..'Inputs_ques'], utils.padQuestion(torch.FloatTensor(q['quesWords'])))
    end
    _G[key..'Inputs_ques'] =  torch.cat(_G[key..'Inputs_ques'], 2):t()
    _G[key..'Inputs_ques_in'] = torch.cat(_G[key..'Inputs_ques_in'], 2):t()
    _G[key..'Inputs_ques_out'] = torch.cat(_G[key..'Inputs_ques_out'], 2):t()
    if (key == 'train') then 
      table.insert(trajectory, {_G['history_'..key]:clone(), train_img:clone(), _G[key..'Inputs_ques_in'], _G[key..'Inputs_ques_out']})
    end
    _G[key..'Inputs'] = {_G[key..'Inputs_ques'], _G[key..'Inputs_spatial'], _G[key..'Inputs_category']}
    if opt.useCuda then
      nn.utils.recursiveType(_G[key..'Inputs'], 'torch.CudaTensor')
    end
    outputs = model_oracle:forward(_G[key..'Inputs'])
    value, index = torch.max(outputs, 2)
    end_count = 0
    for i = 1, opt.batchSize do
        historys[i] = historys[i] or {}
        historyRounds[i] = historyRounds[i] or {}
        historyRounds[i][round] = historyRounds[i][round] or {}
        if (not utils.inTable(endlist, i)) then
          if ( ((torch.FloatTensor(Questions[i]['quesWords']):gt(0):sum(1))[1] > 0)  ) then
              for j, word in ipairs(Questions[i]['quesWords']) do
                  table.insert(historys[i], word)
                  table.insert(historyRounds[i][round], word)
              end
              table.insert(historys[i], wtoi[answers[index[i][1]]])
              table.insert(historyRounds[i][round], wtoi[answers[index[i][1]]])
              _G['history_'..key][i] = utils.padHistory(torch.FloatTensor(historys[i]))
              _G['historyMemory_'..key][i][round] = utils.padMemory(torch.FloatTensor(historyRounds[i][round]))
              end_count = 0
          else
              table.insert(endlist, i)
              if (key == 'train') then end_mask[i] = 1 end
              end_count = end_count + 1
          end
        end
        if (end_count == (opt.batchSize)) then break end
    end
    round = round + 1
  end
end

for epoch=opt.startEpoch, opt.epochs do
    image_list = nil
    img_list = nil
    img_list_shuffle = nil
    trainId2url = nil
    quesSucc_ids_train = nil
    collectgarbage()
    key = "train"
    if opt.debug then
      key = "valid"
    end
    total_count = 0
    image_list = {}
    for line in io.lines('data/'..'guesswhat.'..key..'.jsonl') do
        json_line = cjson.decode(line)
        image_id = json_line['image']['id']
        content = {image = json_line['image'], objects = json_line['objects'] }
        image_list[tostring(image_id)] = content
        total_count = total_count + 1
        if opt.debug then
          if total_count >= 1024 then break end
        end
    end
    collectgarbage()
    img_list = {}
    -- Random generate games for each images
    for i, image in pairs(image_list) do
        j = torch.randperm(#image.objects)[1]
        content = {image = image.image, objects = image.objects, object_id = image.objects[j].id}
        table.insert(img_list, content)
    end

    image_list = nil
    collectgarbage()
    trainSetsize = #img_list
    shuffle = torch.randperm(trainSetsize)
    if opt.verbose then print("#img_list", #img_list) end -- 376972 trian all possible games, 131,394 games used for surpervison, 46794 unique images

    img_list_shuffle = {}

    for i, img in ipairs(img_list) do
      table.insert(img_list_shuffle, img_list[shuffle[i]])
    end
    img_list = nil
    collectgarbage()
    if opt.verbose then print(#img_list_shuffle) end
    quesSucc_ids_train = {}
    trainId2url = {}

    collectgarbage()
    train_imgs = torch.FloatTensor(trainSetsize, opt.imgFeatureSize):fill(0)
    for i, img in ipairs(img_list_shuffle) do
        table.insert(quesSucc_ids_train, i)
        image_fc8 = dataset['img_feat'][id2index[tostring(img.image.id)]]
        train_imgs[i] = image_fc8
    end
    quesSucc_ids_train = torch.FloatTensor(quesSucc_ids_train)
    collectgarbage()
    if opt.verbose then print("loading image done!") end
    trainInputs_spatials = torch.FloatTensor(trainSetsize, 8):fill(0)
    trainInputs_categorys = torch.FloatTensor(trainSetsize):fill(0)
    trainInputs_spatialLists = torch.FloatTensor(trainSetsize, 20, 8):fill(0)
    trainInputs_categoryLists = torch.FloatTensor(trainSetsize, 20):fill(0)
    trainlabels = torch.FloatTensor(trainSetsize)
    for i, img in ipairs(img_list_shuffle) do
      trainInputs_spatials[i] = torch.FloatTensor(utils.get_normalized_spatial_info(img))
      trainInputs_categorys[i] = utils.get_category(img)
      trainInputs_spatialLists[i] = utils.get_normalized_spatial_guesser(img)
      trainInputs_categoryLists[i] = utils.get_objectCategoryList(img)
      trainlabels[i] = utils.get_objectLabel(img)
    end
    collectgarbage()
    for i, img in ipairs(img_list_shuffle) do
      explore_id = utils.split(img['image']['coco_url'], "/")
      trainId2url[tostring(i)] = 'http://cocodataset.org/#explore?id='..explore_id[#explore_id]
    end
    collectgarbage()

    running_average = 0
    for t = 1, trainSetsize, opt.batchSize do
        xlua.progress(torch.floor(t/opt.batchSize)+1, torch.floor(trainSetsize/opt.batchSize))
        history_train, historyMemory_train, train_img, quesSucc_id_train, trainInputs_spatial, trainInputs_category, 
        trainInputs_spatialList, trainInputs_categoryList, label = load( 'train', t )
        round = 1
        end_count= 0
        trajectory = {}
        endlist = {}
        for k,v in ipairs(temperatures) do
          decoder:get(5):get(1):get(1):get(1).temperature = v
          end_mask = torch.FloatTensor(opt.batchSize):fill(0)
          QuesAns( 'train', true )
          end_mask = end_mask:view(end_mask:size(1), 1)
          end_mask = end_mask:expand(opt.batchSize, (opt.beamLen-1))
          trainInputs_ques_in = trainInputs_ques_in:fill(0)
          trainInputs_ques_out = trainInputs_ques_out:fill(0)
          trainInputs_ques_in[{{}, {-1}}] = wtoi['<START>']
          trainInputs_ques_out[{{}, {-1}}] = wtoi['<END>']
          trainInputs_ques_in = trainInputs_ques_in:cmul(end_mask)
          trainInputs_ques_out = trainInputs_ques_out:cmul(end_mask)
          table.insert(trajectory, {history_train:clone(), train_img:clone(), trainInputs_ques_in, trainInputs_ques_out})

          maxLen = trainInputs_categoryList:nonzero()[{{},{2}}]:max(1)[1][1]
          trainInputs_spatialList = trainInputs_spatialList[{{},{1, maxLen}}]
          trainInputs_categoryList = trainInputs_categoryList[{{},{1, maxLen}}]
          
          utils.adjustModel( model_guesser )

          trainInputs_dial = historyMemory_train
          trainInputs = {trainInputs_dial, trainInputs_spatialList, trainInputs_categoryList}
          if opt.useCuda then
            nn.utils.recursiveType(trainInputs, 'torch.CudaTensor')
          end
          model_guesser:evaluate()
          trainInputs[1] = utils.checkHistoryMemory(trainInputs):clone()
          local outputs = model_guesser:forward(trainInputs)
          if opt.useCuda then
              outputs:cuda()
              label:cuda()
          end
          num = 1
          value, index = torch.max(outputs[num], 1)
          if (index[1] == label[1]) then status = "success" else status = "failure" end
          -- if opt.verbose then print(' ') print("dialogue:", utils.itowTable(historys[num]), 
          --   "status", status, "image_url", Questions[num]['image_url']) end
          _, predicts = torch.max(outputs, outputs:dim())
          if opt.useCuda then 
            label = label:type("torch.CudaLongTensor")
          else
            label = label:long()
          end

          reward_correct = torch.eq(predicts, label)
          reward_correct = reward_correct:float()
          reward = reward_correct:clone()
          REWARD = reward:clone()

          model_questioner:training()
          model_questioner:zeroGradParameters()

          model_guesser:training()
          model_guesser:zeroGradParameters()
          local outputs_train = model_guesser:forward(trainInputs)
          reward_target_guesser = REWARD:clone()
          if opt.useCuda then reward_target_guesser = reward_target_guesser:cuda() end
          baseline_target_guesser = reward_target_guesser:clone():fill(running_average)
          -- reward_target_guesser = reward_target_guesser - baseline_target_guesser
          criterion_vrd_guesser:forward(outputs_train, reward_target_guesser)
          local gradCriterionOut_guesser = criterion_vrd_guesser:backward(outputs_train, reward_target_guesser) 
          reward_mask_guesser = torch.FloatTensor(gradCriterionOut_guesser:size()):fill(0)
          for i = 1, label:size(1) do
            reward_mask_guesser[i][label[i]] = 1
          end
          if opt.useCuda then reward_mask_guesser = reward_mask_guesser:cuda() end
          gradCriterionOut_guesser = gradCriterionOut_guesser:cmul(reward_mask_guesser)
          model_guesser:backward(outputs_train, gradCriterionOut_guesser)  
          model_guesserW, model_guesserdW = model_guesser:getParameters()
          if opt.trainGuesser then
            sgd(model_guesserW, model_guesserdW, opt.learningRate * 0.1)
          end

          for i, j in ipairs(trajectory) do 
            history_train = trajectory[i][1]
            train_img = trajectory[i][2]
            quesSucc_in_train = trajectory[i][3]
            quesSucc_out_train = trajectory[i][4]
            inputs = {}
            table.insert(inputs, history_train:t())
            table.insert(inputs, train_img)
            if opt.useCuda then
              nn.utils.recursiveType(inputs, 'torch.CudaTensor')
              model_questioner:cuda()
              criterion_vrd:cuda()
            end
            local encOut = encoder:forward(inputs)
            utils.forwardConnect(encoder, decoder, encOut, history_train:size(2))
            quesSucc_in_train = quesSucc_in_train:t()
            quesSucc_out_train = quesSucc_out_train:t()
            local decOut = decoder:forward(quesSucc_in_train)
            BASELINE = running_average
            reward_target = decOut:sum(3):squeeze():clone()
            reward_target:fill(0)
            baseline_target = reward_target:clone()
            reward_index = decOut:sum(3):squeeze():ne(0):sum(1):squeeze()
            for i = 1, reward_index:size(1) do
              for j = 1, reward_index[i] do
                if REWARD:squeeze()[i] == 1 then
                  reward_target[j][i] = 1 --/ (reward_index[i] - j + 1)
                end
                baseline_target[j][i] = BASELINE --/ (reward_index[i] - j + 1)
              end
            end
            reward_target = reward_target:view(reward_target:size(1), reward_target:size(2), 1)
            reward_target = reward_target - baseline_target
            criterion_vrd:forward(decOut, reward_target)
            local gradCriterionOut = criterion_vrd:backward(decOut, reward_target) -- gradCriterionOut is 0
            reward_mask = torch.FloatTensor(gradCriterionOut:size()):fill(0)
            for i = 1, quesSucc_out_train:size(1) do
              for j = 1, quesSucc_out_train:size(2) do
                if not (quesSucc_out_train[i][j] == 0) then
                  reward_mask[i][j][ quesSucc_out_train[i][j] ] = 1
                end
              end
            end
            if opt.useCuda then reward_mask = reward_mask:cuda() end
            gradCriterionOut = gradCriterionOut:cmul(reward_mask)
            decoder:backward(quesSucc_in_train, gradCriterionOut)
            local gradDecOut = utils.backwardConnect(encoder, decoder)
            encoder:backward(inputs, gradDecOut)
            reward_input = torch.FloatTensor({REWARD:mean()})
            if opt.useCuda then
              reward_input = reward_input:cuda()
            end
          end
        end
        model_questionerW, model_questionerdW = model_questioner:getParameters()
        W_copy = model_questionerW:clone()
        model_questionerdW:clamp(-5, 5)
        sgd(model_questionerW, model_questionerdW, opt.learningRate)
        assert(not(W_copy == model_questionerW))
        model_questioner:zeroGradParameters()
        running_average = ( (t-1)*running_average + (opt.batchSize)*(REWARD:mean()) ) / (t-1+opt.batchSize)
    end
    print("epoch", epoch, "running_average", running_average)
    -- validation accuray and saving model
    running_average = 0
    model_questioner:evaluate()
    opt.batchSize = batchSize
    if opt.debug then
      validSetsize = 1024
    end

    for t = 1, validSetsize, opt.batchSize do
        xlua.progress(torch.floor(t/opt.batchSize)+1, torch.floor(validSetsize/opt.batchSize)) 
        history_valid, historyMemory_valid, valid_img, quesSucc_id_valid, validInputs_spatial, validInputs_category, 
        validInputs_spatialList, validInputs_categoryList, label = load( 'valid', t )
        round = 1
        end_count= 0
        endlist = {}
        QuesAns( 'valid', false )
        maxLen = validInputs_categoryList:nonzero()[{{},{2}}]:max(1)[1][1]
        validInputs_spatialList = validInputs_spatialList[{{},{1, maxLen}}]
        validInputs_categoryList = validInputs_categoryList[{{},{1, maxLen}}]
        
        utils.adjustModel( model_guesser )

        validInputs_dial = historyMemory_valid
        validInputs = {validInputs_dial, validInputs_spatialList, validInputs_categoryList}
        if opt.useCuda then
          nn.utils.recursiveType(validInputs, 'torch.CudaTensor')
        end
        local outputs = model_guesser:forward(validInputs)
        if opt.useCuda then
            outputs:cuda()
            label:cuda()
        end
        num = 1
        value, index = torch.max(outputs[num], 1)
        _, predicts = torch.max(outputs, outputs:dim())
        if opt.useCuda then 
          label = label:type("torch.CudaLongTensor")
        else
          label = label:long()
        end
        reward = torch.eq(predicts, label)
        reward = reward:float()
        if (index[1] == label[1]) then status = "success" else status = "failure" end
        -- if opt.verbose then print(' ') print("dialogue:", utils.itowTable(historys[num]), "status", status, "image_url", Questions[num]['image_url']) end
        running_average = ( (t-1)*running_average + (opt.batchSize)*(reward:mean()) ) / (t-1+opt.batchSize)
        collectgarbage()
    end
    opt.batchSize = batchSize
    print("epoch", epoch, color.fg.BLUE .. "valid running_average", color.fg.BLUE .. tostring(running_average))
    if opt.verbose then print("dialogue:", utils.itowTable(historys[1]), "status", status, "image_url", Questions[1]['image_url']) end

    if opt.bestValid <= running_average then
      opt.bestValid = running_average
      earlyStopCount = 0
      best_valid_model_questioner = model_questioner:clone()
      best_valid_model_guesser = model_guesser:clone()
      best_valid_model_questioner:clearState()
      best_valid_model_guesser:clearState()
      best_valid_model_questioner = best_valid_model_questioner:float()
      best_valid_model_guesser = best_valid_model_guesser:float()
      if opt.saveModel then
        local savepath_questioner = opt.modelPath.."QGen_reinforce.t7"
        local savepath_guesser = opt.modelPath.."Guesser_reinforce.t7"
        torch.save(savepath_questioner, best_valid_model_questioner)
        torch.save(savepath_guesser, best_valid_model_guesser)
        print("Saving model...")
      end
    else
       earlyStopCount = 0 or (earlyStopCount + 1)
    end 

    if earlyStopCount >= opt.earlyStopThresh then
       print("Early stopping at epoch: " .. tostring(epoch))
       break
    end
end
collectgarbage()

quesSucc_ids_test, test_imgs, testInputs_spatials, testInputs_categorys, testInputs_spatialLists, 
testInputs_categoryLists, testlabels, testSetsize, testId2url = json2vec( 'test' )

running_average = 0
model_questioner = best_valid_model_questioner:clone()
model_guesser = best_valid_model_guesser:clone()
if opt.useCuda then 
  model_questioner = model_questioner:cuda() 
  model_guesser = model_guesser:cuda()
end
model_questioner:evaluate()
model_guesser:evaluate()
opt.batchSize = batchSize
if opt.debug then
  testSetsize = 1024
end

for t = 1, testSetsize, opt.batchSize do
    xlua.progress(torch.floor(t/opt.batchSize)+1, torch.floor(testSetsize/opt.batchSize)) 
    history_test, historyMemory_test, test_img, quesSucc_id_test, testInputs_spatial, testInputs_category, 
    testInputs_spatialList, testInputs_categoryList, label = load( 'test', t )
    round = 1
    end_count= 0
    endlist = {}
    QuesAns( 'test', false )
    maxLen = testInputs_categoryList:nonzero()[{{},{2}}]:max(1)[1][1]
    testInputs_spatialList = testInputs_spatialList[{{},{1, maxLen}}]
    testInputs_categoryList = testInputs_categoryList[{{},{1, maxLen}}]

    utils.adjustModel( model_guesser )

    testInputs_dial = historyMemory_test
    testInputs = {testInputs_dial, testInputs_spatialList, testInputs_categoryList}
    if opt.useCuda then
      nn.utils.recursiveType(testInputs, 'torch.CudaTensor')
    end
    local outputs = model_guesser:forward(testInputs)
    if opt.useCuda then
        outputs:cuda()
        label:cuda()
    end
    num = 1
    value, index = torch.max(outputs[num], 1)
    _, predicts = torch.max(outputs, outputs:dim())
    if opt.useCuda then 
      label = label:type("torch.CudaLongTensor")
    else
      label = label:long()
    end
    reward = torch.eq(predicts, label)
    reward = reward:float()
    if (index[1] == label[1]) then status = "success" else status = "failure" end
    -- if opt.verbose then print(' ') print("dialogue:", utils.itowTable(historys[num]), "status", status, "image_url", Questions[num]['image_url']) end
    running_average = ( (t-1)*running_average + (opt.batchSize)*(reward:mean()) ) / (t-1+opt.batchSize)
    collectgarbage()
end
opt.batchSize = batchSize
print(color.fg.RED .. "test running_average", color.fg.RED .. tostring(running_average))

