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

local VarianceReducedReward, parent = torch.class("nn.VarianceReducedReward", "nn.Criterion")

function VarianceReducedReward:__init(module, scale)
   parent.__init(self)
   self.module = module
   self.scale = scale or 1
   self.sizeAverage = true
   self.gradInput = torch.Tensor()
end

function VarianceReducedReward:updateOutput(input, target)
   local input = self:toBatch(input, 1)
   self._reward = target
   self.reward = self.reward or input.new()
   self.reward:resize(self._reward:size(1)):copy(self._reward)
   self.reward:mul(self.scale)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

function VarianceReducedReward:updateGradInput(input, target)
   self._reward = target:clone()
   self._reward:expandAs(self._reward, input)
   self.gradInput = self._reward:clone()
   return self.gradInput
end

function VarianceReducedReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self.__maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
