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

local Reinforce_rewardGradient, parent = torch.class("nn.Reinforce_rewardGradient", "nn.Module")

function Reinforce_rewardGradient:__init(stochastic)
   parent.__init(self)
   self.stochastic = stochastic
end

function Reinforce_rewardGradient:updateOutput(input)
   self.output:set(input)
end

function Reinforce_rewardGradient:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end

function Reinforce_rewardGradient:rewardAs(input)
   if self.reward == nil then 
      self.reward = torch.Tensor(input:size()):fill(0):sum(2):squeeze()
   end
   assert(self.reward:dim() == 1)
   if input:isSameSizeAs(self.reward) then
      return self.reward
   else
      if self.reward:size(1) ~= input:size(1) then
         input = input:sub(1,self.reward:size(1))
         assert(self.reward:size(1) == input:size(1), self.reward:size(1).." ~= "..input:size(1))
      end
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      size[1] = self.reward:size(1)
      self._reward:view(self.reward, table.unpack(size))
      self.__reward:expandAs(self._reward, input)
      return self.__reward
   end
end
