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
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'hdf5'
utils = require 'misc.utils'
require 'xlua'
local cjson = require 'cjson'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-batch_size', 64, 'batch_size')
cmd:option('-out_name', 'data/data_img.h5', 'output name')
opt = cmd:parse(arg)

cutorch.setDevice(1)
net=loadcaffe.load('model/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
    'model/vgg16/VGG_ILSVRC_16_layers.caffemodel', 
    'nn')
net:evaluate()
net=net:cuda()

function loadim(imname)
    im=image.load(imname)
    im=image.scale(im,224,224)
    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end
    im=im*255;
    im2=im:clone()
    im2[{{3},{},{}}]=im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}]=im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}]=im[{{3},{},{}}]-103.939
    return im2
end

img_list = {}
id2index = {}
index = 1
for i, key in pairs({'valid', 'test', 'train'}) do
    for line in io.lines('data/guesswhat.'..key..'.jsonl') do
        json_line = cjson.decode(line)
        image_id = json_line['image']['id']
        file_name = json_line['image']['file_name']
        folder_name = utils.split(file_name, "_")[2]
        image_path = 'data/images/'..folder_name..'/'..file_name
        if not ( utils.inTable(img_list, image_path) ) then
            table.insert(img_list, image_path)
            id2index[tostring(image_id)] = index
            index = index + 1
        end
    end
end
assert(id2index[tostring(489209)] == 1, "Error: key ~= index")
assert(#img_list == 66537, "Error: the number of unique images is wrong")

utils.writeJSON("data/id2index.json", id2index)
id2index = utils.readJSON("data/id2index.json");
assert(id2index[tostring(489209)] == 1, "Error: key ~= index")
print("Wrote image ID to index file 'id2index.json'.")

for i, key in pairs({'valid', 'test', 'train'}) do
    _G[key..'_img'] = {}
    for line in io.lines('data/guesswhat.'..key..'.jsonl') do
        json_line = cjson.decode(line)
        for j, qa in pairs(json_line['qas']) do
            if json_line['status'] == 'success' then
                image_id = json_line['image']['id']
                index = id2index[tostring(image_id)]
                table.insert(_G[key..'_img'], index)
            end
        end
        if json_line['status'] == 'success' then
            table.insert(_G[key..'_img'], index)
        end
    end
end
assert(valid_img[1]==1, "Error: the index is wrong")
assert(#train_img==570078, "Error: the number of train images is wrong")
assert(#valid_img==118484, "Error: the number of valid images is wrong")
assert(#test_img==119778, "Error: the number of test images is wrong")
print("Assertion: OK")

local ndims=4096
local batch_size = opt.batch_size
local sz=#img_list
local img_feat=torch.CudaTensor(sz,ndims)
print(string.format('processing %d images...',sz))
for i=1,sz,batch_size do
    xlua.progress(i+batch_size, sz)
    r=math.min(sz,i+batch_size-1)
    ims=torch.CudaTensor(r-i+1,3,224,224)
    for j=1,r-i+1 do
        ims[j]=loadim(img_list[i+j-1])
    end
    net:forward(ims)
    img_feat[{{i,r},{}}]=net.modules[37].output:clone()
    collectgarbage()
end
print('L2-normalizing image features')
local nm = torch.sqrt(torch.sum(torch.cmul(img_feat, img_feat), 2));
img_feat = torch.cdiv(img_feat, nm:expandAs(img_feat)):float();
collectgarbage()
local h5_file = hdf5.open(opt.out_name, 'w')
h5_file:write('/img_feat', img_feat:float())
h5_file:write('/train_img', torch.Tensor(train_img):float())
h5_file:write('/valid_img', torch.Tensor(valid_img):float())
h5_file:write('/test_img', torch.Tensor(test_img):float())
h5_file:close()
print("Preprocessing images is done!")
