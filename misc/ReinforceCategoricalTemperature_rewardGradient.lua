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

local ReinforceCategoricalTemperature_rewardGradient, parent = torch.class("nn.ReinforceCategoricalTemperature_rewardGradient", "nn.Reinforce_rewardGradient")

function ReinforceCategoricalTemperature_rewardGradient:updateOutput(input)
   self.temperature = self.temperature or 1
   self.stochastic = self.stochastic or false
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaLongTensor() or torch.LongTensor())-- CudaTensor()
   if self.stochastic then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input):add(0.00000001) 
      if type(self.temperature) == 'number' then
         self._input = self._input:div(self.temperature)
      else
         self._input = self._input:cdiv(self.temperature)
      end
      input.multinomial(self._index, self._input, 1)
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   else
      self.output:copy(input)
   end
   return self.output
end

function ReinforceCategoricalTemperature_rewardGradient:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k])  
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput = gradOutput:clone()
   self._input = self._input or input.new()
   self._input:resizeAs(input):copy(input):add(0.00000001) 
   self.gradInput:cdiv(self._input)
   self.gradInput:mul(-1)
   return self.gradInput
end

function ReinforceCategoricalTemperature_rewardGradient:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end
