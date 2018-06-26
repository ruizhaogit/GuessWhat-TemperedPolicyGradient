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

local cjson = require 'cjson'

local utils = {}

function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

function utils.readJSON(fileName)
    local file = io.open(fileName, 'r')
    local text = file:read()
    file:close()
    return cjson.decode(text)
end

function utils.writeJSON(fileName, luaTable)
    local text = cjson.encode(luaTable)
    local file = io.open(fileName, 'w')
    file:write(text)
    file:close()
end

function utils.inTable(tbl, item)
    for key, value in pairs(tbl) do
        if value == item then return key end
    end
    return false
end

function utils.split(pString, pPattern)
   local Table = {}
   local fpat = "(.-)" .. pPattern
   local last_end = 1
   local s, e, cap = pString:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
     table.insert(Table,cap)
      end
      last_end = e+1
      s, e, cap = pString:find(fpat, last_end)
   end
   if last_end <= #pString then
      cap = pString:sub(last_end)
      table.insert(Table, cap)
   end
   return Table
end

function utils.idToWords(vector, ind2word)
    local sentence = ''
    local nextWord
    for wordId = 1, vector:size(1) do
        if vector[wordId] > 0 then
            nextWord = ind2word[tostring(vector[wordId])] 
            sentence = sentence..' '..nextWord
        end
        if nextWord == '<END>' then break end
    end
    
    return sentence
end

function utils.idToQues(vector, ind2word)
    local sentence = ''
    local nextWord
    local question = false
    for wordId = 1, vector:size(1) do
        if (vector[wordId] > 0) then
            nextWord = ind2word[tostring(vector[wordId])] 
            if not question then
              sentence = sentence..' '..nextWord
            end
        end
        if nextWord == '?' then
          question = true
        end
        if nextWord == '<END>' then
          sentence = sentence..' '..nextWord
          break
        end
    end
    return sentence
end

function utils.idToUrl(img_id, key)
  local image_url
  local json_line = 'data/guesswhat.'..key..'.jsonl'
  for line in io.lines(json_line) do
      img = cjson.decode(line)
      if img['id'] == img_id then
        image_url = img['image']['coco_url']
        return image_url
      end
  end
end

function utils.forwardConnect(enc, dec, encOut, seqLen)
    if enc.rnnLayers ~= nil then
        for ii = 1, #enc.rnnLayers do
            dec.rnnLayers[ii].userPrevOutput = enc.rnnLayers[ii].output[seqLen]
            dec.rnnLayers[ii].userPrevCell = enc.rnnLayers[ii].cell[seqLen]
        end
        dec.rnnLayers[#enc.rnnLayers].userPrevOutput = encOut
    else
        dec.rnnLayers[#dec.rnnLayers].userPrevOutput = encOut
    end
end

function utils.backwardConnect(enc, dec)
    if enc.rnnLayers ~= nil then
        for ii = 1, #dec.rnnLayers do
            enc.rnnLayers[ii].userNextGradCell = dec.rnnLayers[ii].userGradPrevCell
            enc.rnnLayers[ii].gradPrevOutput = dec.rnnLayers[ii].userGradPrevOutput
        end
        return dec.rnnLayers[#enc.rnnLayers].userGradPrevOutput
    else
        return dec.rnnLayers[#dec.rnnLayers].userGradPrevOutput
    end
end

function utils.decoderConnect(dec)
    for ii = 1, #dec.rnnLayers do
        dec.rnnLayers[ii].userPrevCell = dec.rnnLayers[ii].cell[1]
        dec.rnnLayers[ii].userPrevOutput = dec.rnnLayers[ii].output[1]
    end
end

function utils.vec2str( vector )
  content = ''
  vector:apply(function(x) 
                if not (x == 0) then 
                  content = content..itow[tostring(x)]..' '
                end
              end)
  return content
end

function utils.pruneToken(vector, word2ind)
    vector = vector[{{2,-1}}]
    local sentence = {};
    local nextWord;
    if (vector:dim()>0) then
      for wordId = 1, vector:size(1) do
          if vector[wordId] > 0 then
              nextWord = vector[wordId] 
              table.insert(sentence, nextWord)
          end
          if (nextWord == word2ind['?']) then break; end
          if (nextWord == word2ind['<END>']) then table.remove(sentence) table.insert(sentence, 0) break; end
      end
    else
      table.insert(sentence, 0)
    end
    return sentence
end

function utils.padQuestion( sentence )
    local pad = sentence:size()[1] - (opt.beamLen - 1)
    sentence = nn.Padding(1, pad, 1, 0):forward(sentence)
    return sentence
end

function utils.padQuestionRight( sentence )
    local pad = sentence:size()[1] - (opt.beamLen - 1)
    sentence = nn.Padding(1, -pad, 1, 0):forward(sentence)
    return sentence
end

function utils.padHistory( sentence )
    local pad = sentence:size()[1] - (opt.beamLen)*(opt.round)
    sentence = nn.Padding(1, pad, 1, 0):forward(sentence)
    return sentence
end

function utils.padMemory( sentence )
    local pad = sentence:size()[1] - 14
    if pad <0 then
        sentence = nn.Padding(1, pad, 1, 0):forward(sentence)
    else 
      sentence = sentence[{{1, 14}}]
      sentence[{14}] = wtoi["?"]
      print("Error: longer than 14")
      os.exit()
    end
    return sentence
end

function utils.greedySearch( history, img, quesSucc_id, key )
  model:evaluate()
  local numQues = history:size(1)
  local startToken = wtoi["<START>"]
  inputs = {}
  table.insert(inputs, history:t())
  table.insert(inputs, img)
  local encOut = encoder:forward(inputs)
  utils.forwardConnect(encoder, decoder, encOut, histLength)
  Questions = {}
  local questionIn = torch.Tensor(1, numQues):fill(startToken)
  local question = {questionIn:t():float()}
  for timeStep = 1, opt.beamLen do
      local decOut = decoder:forward(questionIn):squeeze()
      utils.decoderConnect(decoder)
      local _, nextToken = torch.max(torch.exp(decOut / 1), 2)
      table.insert(question, nextToken:float())
      questionIn:copy(nextToken)
  end
  question = nn.JoinTable(-1):forward(question)
  for iter = 1, numQues do
      local histWords = history[{{iter}, {}}]:squeeze():float()
      local quesWords = question[{{iter}, {}}]:squeeze()
      local histText = utils.idToWords(histWords, itow)
      local quesText = utils.idToQues(quesWords, itow)
      table.insert(Questions, {history = histText, question = quesText, id = quesSucc_id[iter], 
        image_url = utils.idToUrl(quesSucc_id[iter], key) })
  end
  print(' ')
  print("Greedy search generated questions", Questions)
  model:training()
end

function table.clone(org)

  return {table.unpack(org)}
end

function utils.sampling( history, img, quesSucc_id, key, id2url, stochastic)
  local startToken = wtoi["<START>"]
  model_questioner:evaluate()
  if opt.DynamicTPG then
      temper_sampling = nn.ReinforceCategoricalTemperature_rewardGradient()
      temper_sampling.stochastic = true
  end
  if stochastic then
    decoder:get(5):get(1):get(1):get(1).stochastic = true
  else
    decoder:get(5):get(1):get(1):get(1).stochastic = false
  end
  local numQues = history:size(1)
  inputs = {}
  table.insert(inputs, history:t())
  table.insert(inputs, img)
  if opt.useCuda then
    nn.utils.recursiveType(inputs, 'torch.CudaTensor')
    if opt.DynamicTPG then temper_sampling = temper_sampling:cuda() end
  end
  encOut = encoder:forward(inputs)
  utils.forwardConnect(encoder, decoder, encOut, history:size(2))

  local Questions = {}

  local questionIn = torch.FloatTensor(1, numQues):fill(startToken)
  if opt.useCuda then questionIn:cuda() end
  local question = {torch.FloatTensor(1, numQues):fill(startToken):t()}
  if opt.useCuda then
    nn.utils.recursiveType(question, 'torch.CudaTensor')
  end
  for timeStep = 1, opt.beamLen-1 do
      local decOut = decoder:forward(questionIn):squeeze()
      if opt.DynamicTPG then 
        decOut_softmax = decoder:get(4).output:squeeze() 
        value, index = decOut_softmax:max(2)
        index = index:squeeze():float()
      else
        value, index = decoder:get(4).output:max(3)
        index = index:squeeze():float()
      end
      utils.decoderConnect(decoder)

      if (key == 'train') and opt.DynamicTPG then
        tempTensor = index:clone():fill(1)
        for i = 1, index:size(1) do
          tempTensor[i] = ( (VocabCount[index[i]]/DocumCount) * ( (opt.tempMax - opt.tempMin)/(opt.round) ) ) + opt.tempMin
        end
        tempTensor = tempTensor:contiguous():view(tempTensor:size(1), 1):expand(tempTensor:size(1), decOut_softmax:size(2))
        if opt.useCuda then tempTensor = tempTensor:cuda() end
        temper_sampling.temperature = tempTensor:clone()
        decOut_temper = temper_sampling:forward(decOut_softmax)
        _, nextToken = torch.max(torch.exp(decOut_temper), torch.exp(decOut_temper):dim())
      else
        _, nextToken = torch.max(torch.exp(decOut), torch.exp(decOut):dim()) 
      end

      if opt.useCuda then 
        table.insert(question, nextToken:view(opt.batchSize, -1):cuda())
      else
        table.insert(question, nextToken:view(opt.batchSize, -1):float())
      end
      questionIn:copy(nextToken)
  end
  question = nn.JoinTable(-1):forward(question)
  for iter = 1, numQues do
      local histWords = history[{{iter}, {}}]:squeeze()
      quesWords = question[{{iter}, {}}]:squeeze()
      if opt.verbose then
        local histText = utils.idToWords(histWords, itow)
        local quesText = utils.idToQues(quesWords, itow)
        quesWords = utils.pruneToken(quesWords, wtoi)
        table.insert(Questions, {history = histText, question = quesText, quesWords = quesWords, id = quesSucc_id[iter], 
          image_url = id2url[tostring(quesSucc_id[iter])] })
      else
        quesWords = utils.pruneToken(quesWords, wtoi)
        table.insert(Questions, {quesWords = quesWords})
      end

      if (key == 'train') and opt.DynamicTPG then
        for i, w in pairs(quesWords) do
          if not (w==0) then
            VocabCount[w] = (VocabCount[w] + 1)
          end
        end
      end

  end
  return Questions
end

function utils.get_bbox( img, object_id )
    for i, ob in pairs(img['objects']) do
        if object_id == ob['id'] then
            return ob['bbox']
        end
    end
end

function utils.get_normalized_spatial_info( img )
    w = img["image"]["width"]
    h = img["image"]["height"]
    object_id = img["object_id"]
    bbox = utils.get_bbox(img, object_id)
    x_min = bbox[1]
    y_min = bbox[2]
    w_box = bbox[3]
    h_box = bbox[4]
    y_max = y_min + h_box
    x_max = x_min + w_box
    x_cen = (x_min + x_max)/2
    y_cen = (y_min + y_max)/2
    normalized_spatial = {(x_min/w)*2-1, (y_min/h)*2-1, (x_max/w)*2-1, (y_max/h)*2-1, 
    (x_cen/w)*2-1, (y_cen/h)*2-1, (w_box/w)*2, (h_box/h)*2}
    return normalized_spatial
end

function utils.get_category( img )
    category_id = 0
    object_id = img["object_id"]
    for i, ob in pairs(img['objects']) do
        if object_id == ob["id"] then
            category_id = ob["category_id"]
        end
    end
    return category_id
end

function utils.get_normalized_spatial_guesser( img )
    n = 20
    max_length = 8
    label_arrays = torch.FloatTensor(n, max_length):fill(0)
    w = img["image"]["width"]
    h = img["image"]["height"]
    for i, ob in ipairs(img["objects"]) do
        bbox = ob["bbox"]
        x_min = bbox[1]
        y_min = bbox[2]
        w_box = bbox[3]
        h_box = bbox[4]
        y_max = y_min + h_box
        x_max = x_min + w_box
        x_cen = (x_min + x_max)/2
        y_cen = (y_min + y_max)/2
        normalized_spatial = {(x_min/w)*2-1, (y_min/h)*2-1, (x_max/w)*2-1, (y_max/h)*2-1, (x_cen/w)*2-1, 
        (y_cen/h)*2-1, (w_box/w)*2, (h_box/h)*2}
        label_arrays[i] = torch.FloatTensor(normalized_spatial)
    end
    return label_arrays
end

function utils.get_objectCategoryList( img )
    n = 20
    label_arrays = torch.FloatTensor(n):fill(0)
    for i, ob in ipairs(img['objects']) do
        label_arrays[i] = ob['category_id']
    end
    return label_arrays
end

function utils.get_objectLabel( img )
    label = 0
    for i, ob in ipairs(img["objects"]) do
        if ob["id"] == img["object_id"] then
            label = i
        end
    end
    if not ((label >= 1) and (label <= 20)) then
        print("Object label out of range!")
        os.exit()
    end
    return label
end

function utils.has_value (tab, val)
    for index, value in ipairs(tab) do
        if value == val then
            return true
        end
    end

    return false
end

function utils.right_align(seq,lengths)
    local v=seq:clone():fill(0)
    local N=seq:size(2)
    for i=1,seq:size(1) do
        v[i][{{N-lengths[i]+1,N}}]=seq[i][{{1,lengths[i]}}]
    end
    return v
end

function utils.itowTable( history )
    new = ' '
    for i, w in ipairs(history) do
      if itow[tostring(w)] then
        new = new..itow[tostring(w)]..' '
      end
    end
    return new
end

function utils.quesToIn( question )
    questionIn = table.clone(question)
    table.insert(questionIn, 1, wtoi["<START>"])
    table.remove(questionIn)
    return questionIn
end

function utils.purn_table( test )
  local hash = {}
  local res = {}
  for _,v in ipairs(test) do
     if (not hash[v]) then
         res[#res+1] = v
         hash[v] = true
     end
  end
  return res
end

function utils.check_ques( trainInputs )
  local pass = true
  local zero_flag = false
  local ques_batch = trainInputs[1]
  for i = 1, ques_batch:size(1) do
    ques_batch[i]:apply(function(x) 
                if not (x == 0) then 
                  zero_flag = true
                end
                if zero_flag then
                  if not ( (x == 0) or (x == wtoi['<START>']) or (x == wtoi['<END>']) ) then
                    pass = pass
                  else
                    pass = false
                    -- print(ques_batch[i])
                  end
                end
              end)
    zero_flag = false
  end
  return pass
end

function utils.adjustModel( model )
  for k, v in ipairs(model.forwardnodes) do
    if v.data.annotations.name == 'replicate' then
      v.data.module.nfeatures = maxLen
    end
  end
  for k, v in ipairs(model.forwardnodes) do
    if v.data.annotations.name == 'view' then
      v.data.module:resetSize(-1, maxLen, 1)
    end
    if v.data.annotations.name == 'view_256' then
      v.data.module:resetSize(-1, maxLen, opt.hiddenSize/2)
    end
  end  
end
function utils.checkHistoryMemory( Inputs )
  local histMemo = Inputs[1]
  assert(histMemo:size(1) == opt.batchSize, 'utils.checkHistoryMemory: size 1 wrong')
  assert(histMemo:size(2) == 14, 'utils.checkHistoryMemory: size 1 wrong')
  assert(histMemo:size(3) == 14, 'utils.checkHistoryMemory: size 1 wrong')

  histMemo:apply(
    function(x)
      if not ( (x <= vocabulary_size_ques) and (x >= 0) ) then
        print('utils.checkHistoryMemory: content wrong: ', x)
        x = 0
      end
      return x
    end)

  return histMemo
end

return utils
