'''
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
'''

import argparse
import numpy as np
import h5py
from nltk.tokenize import word_tokenize , sent_tokenize
import json
import json_lines
import sys
from tqdm import tqdm
execfile("misc/utils.py")
reload(sys)  
sys.setdefaultencoding('utf8')

parser = argparse.ArgumentParser()
parser.add_argument('--input_train_json', default='data/guesswhat.train.jsonl', help='input jsonl file')
parser.add_argument('--input_valid_json', default='data/guesswhat.valid.jsonl', help='input jsonl file')
parser.add_argument('--input_test_json', default='data/guesswhat.test.jsonl', help='input jsonl file')
parser.add_argument('--output_json', default='data/vocab.json', help='output json file')
parser.add_argument('--output_h5', default='data/data_qas.h5', help='output h5 file')
parser.add_argument('--word_count_threshold', default=2, type=int, help='only words that occur more than this number of times will be put in vocab')
parser.add_argument('--max_rounds', default=14, type=int, help='14 is the max number of rounds in the 98 percent of the data')
parser.add_argument('--max_length', default=13, type=int, help='13 is the max length of quesitons in the 98 percent of the data')

args = parser.parse_args()
params = vars(args)
print('Input parameters:')
print(json.dumps(params, indent = 2))

for key in ['train', 'valid', 'test']:
    vars()["imgs_"+key] = jsonLines_list(params['input_'+key+'_json']) 
    print("tokenizing "+key+"set")
    vars()["imgs_"+key] = tokenize_txt(vars()["imgs_"+key], params)

vocab = build_vocab(imgs_train, params)
itow = {i+1:w for i,w in enumerate(vocab)} 
wtoi = {w:i+1 for i,w in enumerate(vocab)} 
vocab.extend(['<START>', '<END>'])
itow[len(wtoi)+1] = '<START>'
itow[len(wtoi)+2] = '<END>'
wtoi['<START>'] = len(itow) - 1
wtoi['<END>'] = len(itow)
out = {}
out['vocab']= vocab
out['itow'] = itow
out['wtoi'] = wtoi
json.dump(out, open(params['output_json'], 'w'))
print("Wrote the vocabulary to " + params['output_json'] + ", which has "+ str(len(vocab)) + " words.")

output_json = json.load(open(params['output_json'], 'r'))
vocab = output_json['vocab']
itow = output_json['itow']
wtoi = output_json['wtoi']

answers = [u'No', u'Yes', u'N/A']
atoi = {w:i+1 for i,w in enumerate(answers)}
itoa = {i+1:w for i,w in enumerate(answers)}

for key in ['train', 'valid', 'test']:
    vars()["imgs_"+key] = apply_vocab(vars()["imgs_"+key], wtoi, params)
    vars()["ques_"+key], vars()["ques_length_"+key], vars()["question_id_"+key] \
    = encode_question_oracle(vars()["imgs_"+key], params, wtoi)
    vars()["answers_"+key] = encode_answer(vars()["imgs_"+key], atoi)
    vars()["spatial_"+key] = encode_normalized_spatial(vars()["imgs_"+key])
    vars()["category_"+key] = encode_category(vars()["imgs_"+key])
    vars()["dial_"+key], vars()["dial_length_"+key] = encode_dialogue(vars()["imgs_"+key], params, wtoi) 
    vars()["spatialList_"+key] = encode_normalized_spatial_list(vars()["imgs_"+key]) 
    vars()["categoryList_"+key] = encode_category_list(vars()["imgs_"+key]) 
    vars()["objectLabel_"+key] = encode_target_object(vars()["imgs_"+key]) 
    vars()["dial_rounds_"+key] = encode_dialogue_rounds_right_aligned(vars()["imgs_"+key], params, wtoi)
    vars()["history_"+key], vars()["history_length_"+key] = encode_history(vars()["imgs_"+key], params, wtoi)
    vars()["ques_"+key] = right_align(vars()["ques_"+key], vars()["ques_length_"+key])
    vars()["dial_"+key] = right_align(vars()["dial_"+key], vars()["dial_length_"+key])
    vars()["history_"+key] = right_align(vars()["history_"+key], vars()["history_length_"+key])
    vars()["quesSucc_"+key], vars()["quesSucc_length_"+key], vars()["quesSucc_id_"+key] =\
     encode_question_qgen(vars()["imgs_"+key], params, wtoi)
    vars()["quesSucc_in_"+key] = np.insert(vars()["quesSucc_"+key], 0, output_json['wtoi']['<START>'], axis=1)
    vars()["quesSucc_in_length_"+key] = new_length(vars()["quesSucc_length_"+key])
    vars()["quesSucc_in_"+key] = cut_ques_token(vars()["quesSucc_in_"+key], vars()["quesSucc_in_length_"+key])
    vars()["quesSucc_in_"+key] = vars()["quesSucc_in_"+key][:, 0:-1]
    vars()["quesSucc_out_"+key] = vars()["quesSucc_"+key]
    vars()["history_"+key], vars()["quesSucc_in_"+key], vars()["quesSucc_out_"+key], vars()["quesSucc_id_"+key] = \
    add_end_token(vars()["history_"+key], vars()["dial_"+key], vars()["quesSucc_in_"+key], \
        vars()["quesSucc_out_"+key], vars()["quesSucc_id_"+key], params)

f = h5py.File(params['output_h5'], "w")
for key in ['train', 'valid', 'test']:
    f.create_dataset("ques_"+key, dtype='uint32', data=vars()["ques_"+key])
    f.create_dataset("answers_"+key, dtype='uint32', data=vars()["answers_"+key])
    f.create_dataset("spatial_"+key, dtype='float32', data=vars()["spatial_"+key])
    f.create_dataset("category_"+key, dtype='float32', data=vars()["category_"+key])
    f.create_dataset("spatialList_"+key, dtype='float32', data=vars()["spatialList_"+key])
    f.create_dataset("categoryList_"+key, dtype='float32', data=vars()["categoryList_"+key])
    f.create_dataset("objectLabel_"+key, dtype='float32', data=vars()["objectLabel_"+key])
    f.create_dataset("dial_rounds_"+key, dtype='uint32', data=vars()["dial_rounds_"+key])
    f.create_dataset("history_"+key, dtype='uint32', data=vars()["history_"+key])
    f.create_dataset("quesSucc_in_"+key, dtype='uint32', data=vars()["quesSucc_in_"+key])
    f.create_dataset("quesSucc_out_"+key, dtype='uint32', data=vars()["quesSucc_out_"+key])
    f.create_dataset("quesSucc_id_"+key, dtype='uint32', data=vars()["quesSucc_id_"+key])
f.close()
print('Wrote preprocessed data to '+params['output_h5']+".")