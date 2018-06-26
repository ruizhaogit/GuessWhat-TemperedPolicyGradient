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

category_size = 90
-- trainInputs = {trainInputs_dial, trainInputs_spatialList, trainInputs_categoryList}
inputs = {}
outputs = {}

table.insert(inputs, nn.Identity()()) -- dial
table.insert(inputs, nn.Identity()()) -- spatialList
table.insert(inputs, nn.Identity()()) -- categoryList

dial = inputs[1]
spatialList = inputs[2]
categoryList = inputs[3]

rnn = nn.SeqLSTM(opt.lookupDim, opt.hiddenSize)
rnn.maskzero = true 

mlpSpatial = nn.Sequential()
mlpSpatial:add(nn.Linear(8, opt.hiddenSize/2))
mlpSpatial:add(nn.ReLU())
mlpSpatial:add(nn.Reshape(1, opt.hiddenSize/2, true))

mlpCategory =  nn.Sequential()
mlpCategory:add(nn.LookupTableMaskZero(category_size, opt.lookupDim))
mlpCategory:add(nn.ReLU())
mlpCategory:add(nn.View(-1,opt.lookupDim))
mlpCategory:add(nn.Linear(opt.lookupDim, opt.hiddenSize/2))
mlpCategory:add(nn.ReLU())
mlpCategory:add(nn.Reshape(1, opt.hiddenSize/2), true)

mlp = nn.Sequential() --3x256
mlp:add(nn.Linear(opt.hiddenSize, opt.hiddenSize))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(opt.hiddenSize, opt.hiddenSize))
mlp:add(nn.ReLU())
split_mlp = nn.Sequential()--3x512
split_mlp:add(nn.Reshape(2, opt.hiddenSize), true) --3x2x256
split_mlp:add(nn.SplitTable(1,2)) --{3x256, 3x256}
split_mlp:add(nn.ParallelTable():add(nn.Identity()):add(mlp)) --3x512
split_mlp:add(nn.JoinTable(2)) --3x512
split_mlp:add(nn.Reshape(1, opt.hiddenSize*2), true) --3x1x512

dot = nn.Sequential() --3x512
dot:add(nn.Reshape(2, opt.hiddenSize)) --3x2x256
dot:add(nn.SplitTable(1,2))
dot:add(nn.DotProduct())
dot:add(nn.View(-1, 1))

mlp2 = nn.Sequential() --3x256
mlp2:add(nn.Linear(opt.hiddenSize, opt.hiddenSize))
mlp2:add(nn.ReLU())
mlp2:add(nn.Linear(opt.hiddenSize, opt.hiddenSize))
mlp2:add(nn.ReLU())

if opt.dropout > 0 then
  model_dial_rounds =  
    dial
  - nn.View(-1, 14)
  - nn.Transpose({1,2})
  - nn.LookupTableMaskZero(vocabulary_size_ques, opt.lookupDim)
  - nn.Dropout(opt.dropout)
  - rnn
  - nn.Dropout(opt.dropout)
  - nn.Select(1, -1)
  - nn.View(-1, 14, opt.hiddenSize)
else
  model_dial_rounds =  
    dial
  - nn.View(-1, 14)
  - nn.Transpose({1,2})
  - nn.LookupTableMaskZero(vocabulary_size_ques, opt.lookupDim)
  - rnn
  - nn.Select(1, -1)
  - nn.View(-1, 14, opt.hiddenSize)
end

mask =
  dial
- nn.Sum(3)
- nn.Clamp(0, 1)
- nn.AddConstant(-1)
- nn.MulConstant(-1)

model_mlpSpatial =    
  spatialList
- nn.SplitTable(1, 2)
- nn.MapTable(mlpSpatial)
- nn.JoinTable(1, 2)

model_view =
  categoryList
- nn.Contiguous()
- nn.View(-1, 20, 1)

model_view:annotate{name = 'view'}

model_mlpCategory =
  model_view
- nn.SplitTable(1, 2)
- nn.MapTable(mlpCategory)
- nn.JoinTable(1, 2)

spatial_sum =
  model_mlpSpatial
- nn.Sum(2)

category_sum =
  model_mlpCategory
- nn.Sum(2)

query =
  { spatial_sum, category_sum }
- nn.JoinTable(2)
- mlp:clone('weight', 'bias', 'gradWeight', 'gradBias')
- mlp2:clone('weight', 'bias', 'gradWeight', 'gradBias')
- nn.View(-1, opt.hiddenSize, 1)

model_dial_rounds_attend =
  model_dial_rounds
- nn.View(-1, opt.hiddenSize)
- nn.Linear(opt.hiddenSize, opt.hiddenSize)
- nn.ReLU()
- nn.Linear(opt.hiddenSize, opt.hiddenSize)
- nn.ReLU()
- nn.View(-1, 14, opt.hiddenSize)

attention =
  {model_dial_rounds_attend, query}
- nn.MM()
- nn.MulConstant(1/torch.sqrt(14))
- nn.Squeeze()

attention_mask =
  {attention, mask}
- nn.MaskSoftMax()
- nn.View(-1, 14, 1)
- nn.View(-1, 1, 1)

dial_attend =
  {nn.View(-1, opt.hiddenSize, 1)(model_dial_rounds), attention_mask}
- nn.MM()
- nn.Squeeze()
- nn.View(-1, 14, opt.hiddenSize)

output_dial =
  dial_attend
- nn.Sum(2)

model_mask =
  categoryList
- nn.Clamp(0, 1)
- nn.AddConstant(-1)
- nn.MulConstant(-1)
- nn.MulConstant(-9999999)

query_2 =
  {output_dial,nn.Linear(opt.hiddenSize, opt.hiddenSize)(nn.View(-1, opt.hiddenSize)(query))}
- nn.CAddTable()

attention_2 =
  {model_dial_rounds_attend, nn.View(-1, opt.hiddenSize, 1)(query_2)}
- nn.MM()
- nn.MulConstant(1/torch.sqrt(14))
- nn.Squeeze()

attention_mask_2 =
  {attention_2, mask}
- nn.MaskSoftMax()
- nn.View(-1, 14, 1)
- nn.View(-1, 1, 1)

dial_attend_2 =
  {nn.View(-1, opt.hiddenSize, 1)(model_dial_rounds), attention_mask_2}
- nn.MM()
- nn.Squeeze()
- nn.View(-1, 14, opt.hiddenSize)

output_dial_2 =
  dial_attend_2
- nn.Sum(2)

model_dial_2 =
  output_dial_2
- nn.Replicate(20, 2)
model_dial_2:annotate{name = 'replicate'}

model_main_2 = 
  { model_dial_2, model_mlpSpatial, model_mlpCategory }
- nn.JoinTable(3)
- nn.SplitTable(1,2) --3x20x512
- nn.MapTable(split_mlp:clone('weight', 'bias', 'gradWeight', 'gradBias'))
- nn.JoinTable(2) --3x20x512
- nn.SplitTable(1,2)
- nn.MapTable(dot) 
- nn.JoinTable(2)