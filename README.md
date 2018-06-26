# GuessWhat-TemperedPolicyGradient

The original GuessWhat?! Game repo is based on TensorFlow and available at:  
https://github.com/GuessWhatGame/guesswhat.  

This repo is based on Torch7 (Lua) and improves the performance by 14% using advanced RNN structures and Tempered Policy Gradient.  

For more details, please refer to our paper "Improving Goal-Oriented Visual Dialog Agents via Advanced Recurrent Nets with Tempered Policy Gradient".  

The code was developed by Rui Zhao (Siemens AG & Ludwig Maximilian University of Munich).  
The implementation is tested on Ubuntu 14.04 using a single GPU with 12GB memory.  

1. Installation:  

- Our code is implemented in [Torch][1] (Lua). Installation instructions are as follows:

```
git clone https://github.com/ruizhaogit/GuessWhat-TemperedPolicyGradient.git
```

- The data preprocessing python script needs the packages including: numpy, h5py, nltk, json_lines, and tqdm.
To install these packages, you can use the "pip" tool:

```
pip install numpy
pip install h5py
pip install nltk
pip install json_lines
pip install tqdm
``` 

To use the tokenizer in nltk, you need download the necessary packages in python:
```
import nltk
nltk.download('punkt')
```

- The code uses the following packages: [torch/torch7][2], [torch/nn][3], [torch/nngraph][4], [Element-Research/dp][15], [Element-Research/rnn][5], [torch/image][6], [lua-cjson][7], [loadcaffe][8], [torch-hdf5][9]. After Torch is installed ([instructions][14]), these can be installed/updated using:

```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc

sudo apt-get install libprotobuf-dev protobuf-compiler
luarocks install loadcaffe
luarocks install dp
luarocks install nngraph
luarocks install lua-cjson

git clone https://github.com/Element-Research/rnn.git
cd rnn
luarocks make rocks/rnn-scm-1.rockspec

luarocks install luabitop
sudo apt-get install libhdf5-serial-dev hdf5-tools
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"
```

Installation instructions for torch-hdf5 are given [here][9].

2. Download datasets
- [GuessWhat?! game][10]:  
```
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz -P data/ 
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz -P data/  
wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz -P data/  
gunzip data/*.jsonl.gz
```  

- [COCO image][11]:  
```
wget http://images.cocodataset.org/zips/train2014.zip -P data/  
wget http://images.cocodataset.org/zips/val2014.zip -P data/  
unzip 'data/*.zip' -d data/images/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'  
rm data/*.zip  
```

3. Preprocess data
- Use [VGG model][12] to extract image features:
```
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt -P model/vgg16/  
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -P model/vgg16/
th data/preprocessImages.lua
```

- Preprocessing the GuessWhat?! dataset, including parsing QAs using [NLTK][13], building the dictionary, calculating the spatial features, encoding the category information:
```
python2.7 data/preprocessQAs.py
```

4. Train and evaluate the models:
- To train the Oracle, Guesser, and the QGen (Question-Generator) model, simply run the lua scripts separately:
```
th oracle.lua --learningRate 0.0001 --epochs 30 --earlyStopThresh 10 --learningRateDecay 1e-7
th guesser.lua --learningRate 0.0001 --epochs 30 --earlyStopThresh 10 --learningRateDecay 1e-7
th qgen.lua --learningRate 0.001 --epochs 30 --earlyStopThresh 10 --learningRateDecay 0.5
```
- For the reinforcement learning part, if you want to use the standard REINFORCE method, please run:
```
th reinforce.lua --temperatures '{1.0}' --epochs 80 --earlyStopThresh 20 --learningRate 0.001
```
- If you want to use Single-TPG, please run:
```
th reinforce.lua --temperatures '{1.5}' --epochs 80 --earlyStopThresh 20 --learningRate 0.001
```
- If you want to use Parallel-TPG, please run:
```
th reinforce.lua --temperatures '{1.0, 1.5}' --epochs 80 --earlyStopThresh 20 --learningRate 0.001
```
- At last, if you want to use the Dynamic-TPG, please run:
```
th reinforce.lua --DynamicTPG --tempMin 0.5 --tempMax 1.5 --epochs 80 --earlyStopThresh 20 --learningRate 0.001
```
- After training, we obtained the following results:

| Method| Accuracy  |
| --------  |:-----:|
| REINFORCE   | 69.66% |
| Single-TPG     | 69.76% |
| Parallel-TPG   | 73.86%|
| Dynamic-TPG   | 74.31%| 


4. Citation:
```
@inproceedings{zhao2018temperedpg,
  title={Improving Goal-Oriented Visual Dialog Agents via Advanced Recurrent Nets with Tempered Policy Gradient},
  author={Zhao, Rui and Tresp, Volker},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI Workshop)},
  year={2018}
}
```

[1]: http://torch.ch/
[2]: https://github.com/torch/torch7
[3]: https://github.com/torch/nn
[4]: https://github.com/torch/nngraph
[5]: https://github.com/Element-Research/rnn/
[6]: https://github.com/torch/image
[7]: https://luarocks.org/modules/luarocks/lua-cjson
[8]: https://github.com/szagoruyko/loadcaffe
[9]: https://github.com/deepmind/torch-hdf5
[10]: https://guesswhat.ai 
[11]: http://cocodataset.org 
[12]: https://gist.github.com/ksimonyan/211839e770f7b538e2d8/
[13]: http://www.nltk.org/
[14]: http://torch.ch/docs/getting-started.html#_
[15]: https://github.com/nicholas-leonard/dp