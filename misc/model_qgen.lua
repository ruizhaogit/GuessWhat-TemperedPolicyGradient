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

nn.FastLSTM.usenngraph = true
enc = nn.Sequential()
concat = nn.ConcatTable()
enc.wordEmbed = nn.LookupTableMaskZero(vocabulary_size, opt.lookupDim)
wordBranch = nn.Sequential()
:add(nn.SelectTable(1)):add(enc.wordEmbed)

enc.rnnLayers = {}
enc.rnnLayers[1] = nn.SeqLSTM(opt.lookupDim, opt.hiddenSize)
enc.rnnLayers[1]:maskZero()
wordBranch:add(enc.rnnLayers[1])
wordBranch:add(nn.Select(1, -1))

concat:add(wordBranch)
concat:add(nn.SelectTable(2))
enc:add(concat)

enc:add(nn.JoinTable(2))
if opt.dropout > 0 then
    enc:add(nn.Dropout(opt.dropout))
end
enc:add(nn.Linear(opt.hiddenSize + opt.imgFeatureSize, opt.hiddenSize))
enc:add(nn.Tanh())

dec = nn.Sequential()
embedNet = enc.wordEmbed:clone('weight', 'bias', 'gradWeight', 'gradBias')
dec:add(embedNet)

dec.rnnLayers = {}
dec.rnnLayers[1] = nn.SeqLSTM(opt.lookupDim, opt.hiddenSize)
dec.rnnLayers[1]:maskZero()
dec:add(dec.rnnLayers[1])
dec:add(nn.Sequencer(nn.MaskZero(nn.Linear(opt.hiddenSize, vocabulary_size), 1)))