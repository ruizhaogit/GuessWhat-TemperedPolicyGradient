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

def jsonLines_list(file_name):
    jsonlist = []
    with open(file_name, 'rb') as f:
        for item in json_lines.reader(f):
            jsonlist.append(item)
    return jsonlist

def sent_word_tokenize(s):
    txt = []
    sents = sent_tokenize(s)
    for i, sent in enumerate(sents):
        txt.extend( word_tokenize(sent.encode('utf-8').lower()) )
    return txt

def tokenize_txt(imgs, params):
    max_length = params['max_length']
    for i,img in enumerate(tqdm(imgs)):
        dialogue_token = []
        for j, qa in enumerate(img["qas"]):
            token_ques = sent_word_tokenize(qa["question"].encode('utf-8').lower())
            if len(token_ques) > max_length:
                token_ques = token_ques[0: max_length]
                token_ques[-1] = '?'
            token_ans = sent_word_tokenize(qa["answer"].encode('utf-8').lower())
            token_dial = token_ques + token_ans
            img['qas'][j]['question_tokens'] = token_ques
            img['qas'][j]['answer_tokens'] = token_ans
            img['qas'][j]['history_tokens'] = dialogue_token
            if j < params['max_rounds']:
                dialogue_token = dialogue_token + token_dial
        img['dialogue_tokens'] = dialogue_token
    return imgs

def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']
    counts = {}
    for img in imgs:
        for w in img['dialogue_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    total_words = sum(counts.itervalues())
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    vocab.append('UNK')
    return vocab

def apply_vocab(imgs, wtoi, params):
    for i, img in enumerate(imgs):
        img['final_dialogue'] = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in img['dialogue_tokens']]
        for j, qa in enumerate(img["qas"]):
            qa['final_question'] = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in qa['question_tokens']]
            qa['final_history'] = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in qa['history_tokens']]
    return imgs

def length_imgs(imgs):
    l = 0
    for img in imgs:
        for qa in img["qas"]:
            l = l + 1
    return l

def encode_question_oracle(imgs, params, wtoi):
    max_length = params['max_length']
    N = length_imgs(imgs)
    arrays = np.zeros((N, max_length), dtype='uint32')
    length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i,img in enumerate(imgs):
        for j, qa in enumerate(img['qas']):
            question_id[question_counter] = qa['id']
            length[question_counter] = min(max_length, len(qa['final_question']))
            for k, w in enumerate(qa['final_question']):
                arrays[question_counter,k] = wtoi[w]
            question_counter += 1
    return arrays, length, question_id

def encode_answer(imgs, atoi):
    N = length_imgs(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')
    counter = 0
    for i, img in enumerate(imgs):
        for j, qa in enumerate(img['qas']):
            ans_arrays[counter] = atoi[qa['answer']]
            counter = counter + 1
    return ans_arrays

def get_bbox(img, object_id):
    for ob in img['objects']:
        if object_id == ob["id"]:
            return ob["bbox"]

def bbox_to_normalized_spatial(bbox, w, h):
    x_min = bbox[0]
    y_min = bbox[1]
    w_box = bbox[2]
    h_box = bbox[3]
    y_max = y_min + h_box
    x_max = x_min + w_box
    x_cen = (x_min + x_max)/2
    y_cen = (y_min + y_max)/2
    normalized_spatial = [(x_min/w)*2-1, (y_min/h)*2-1, (x_max/w)*2-1, (y_max/h)*2-1, 
        (x_cen/w)*2-1, (y_cen/h)*2-1, (w_box/w)*2, (h_box/h)*2]
    return normalized_spatial

def get_normalized_spatial(img):
    w = img["image"]["width"]
    h = img["image"]["height"]
    object_id = img["object_id"]
    bbox = get_bbox(img, object_id)
    normalized_spatial = bbox_to_normalized_spatial(bbox, w, h)
    return normalized_spatial

def encode_normalized_spatial(imgs):
    N = length_imgs(imgs)
    max_length = 8
    arrays = np.zeros((N, max_length), dtype='float32')
    counter = 0
    for i,img in enumerate(imgs):
        for j, qa in enumerate(img['qas']):
            spatial = get_normalized_spatial(img)
            for k,n in enumerate(spatial):
                arrays[counter,k] = n           
            counter = counter + 1
    return arrays

def get_category(img):
    max_length = 90
    arrays = 0
    object_id = img["object_id"]
    for ob in img['objects']:
        if object_id == ob["id"]:
            category_id = ob["category_id"]
            arrays = category_id
    return arrays

def encode_category(imgs):
    N = length_imgs(imgs)
    arrays = np.zeros(N, dtype='float32')
    counter = 0
    for i,img in enumerate(imgs):
        for j, qa in enumerate(img['qas']):
            category = get_category(img)
            arrays[counter] = category           
            counter = counter + 1
    return arrays

def get_success_num_img(imgs):
    counter = 0
    for i, img in enumerate(imgs):
        if img['status'] == 'success': # 'incomplete' or 'failure' or 'success'
            counter = counter + 1
    return counter

def get_success_num_qas(imgs):
    l = 0
    for img in imgs:
        if img['status'] == 'success':
            for qa in img["qas"]:
                l = l + 1
    return l

def encode_dialogue(imgs, params, wtoi):
    max_length = ( params['max_length'] + 1 ) * params['max_rounds']
    N = get_success_num_img(imgs)
    counter = 0
    arrays = np.zeros((N, max_length), dtype='uint32')
    length = np.zeros(N, dtype='uint32')
    for i,img in enumerate(imgs):
        if img['status'] == 'success':
            length[counter] = min(max_length, len(img['final_dialogue']))
            for k,w in enumerate(img['final_dialogue']):
                arrays[counter,k] = wtoi[w]
            counter = counter + 1
    return arrays, length

def encode_dialogue_rounds_right_aligned(imgs, params, wtoi):
    counter = 0
    N = get_success_num_img(imgs)
    arrays = np.zeros((N, params['max_rounds'], (params['max_length']+1)), dtype='uint32')
    for i, img in enumerate(imgs):
        if img['status'] == 'success':
            qa_token = []
            for r, qa in enumerate(img['qas']):
                if r < params['max_rounds']:
                  q_token = img['qas'][r]['question_tokens']
                  a_token = img['qas'][r]['answer_tokens']
                  qa_token = q_token + a_token
                  l = 0
                  for j, w in reversed(list(enumerate(qa_token))):
                      try:
                          arrays[counter, r, (params['max_length'] - l)] = wtoi[w]
                      except:
                          arrays[counter, r, (params['max_length'] - l)] = wtoi['UNK']
                      l = l + 1
            counter = counter + 1
    return arrays

def get_normalized_spatial_list(img):
    n = 20
    max_length = 8
    arrays = np.zeros((n, max_length), dtype='float32')
    w = img["image"]["width"]
    h = img["image"]["height"]
    for i, ob in enumerate(img["objects"]):
        bbox = ob["bbox"]
        normalized_spatial = bbox_to_normalized_spatial(bbox, w, h)
        arrays[i] = normalized_spatial
    return arrays

def encode_normalized_spatial_list(imgs):
    N = get_success_num_img(imgs)
    n = 20
    max_length = 8
    counter = 0
    arrays = np.zeros((N, n, max_length), dtype='float32')
    for i, img in enumerate(imgs):
        if img['status'] == 'success':
           arrays[counter] = get_normalized_spatial_list(img)
           counter = counter + 1
    return arrays

def get_category_list(img):
    n = 20
    arrays = np.zeros(n, dtype='float32')
    for i, ob in enumerate(img['objects']):
        arrays[i] = ob['category_id']
    return arrays

def encode_category_list(imgs):
    N = get_success_num_img(imgs)
    n = 20
    counter = 0
    arrays = np.zeros((N, n), dtype='float32')
    for i, img in enumerate(imgs):
        if img['status'] == 'success':
           arrays[counter] = get_category_list(img)
           counter = counter + 1
    return arrays

def get_target_object(img):
    label = 0
    for i, ob in enumerate(img["objects"]):
        if ob["id"] == img["object_id"]:
            label = i
    return label + 1 #for torch

def encode_target_object(imgs):
    N = get_success_num_img(imgs)
    arrays = np.zeros(N, dtype='float32')
    counter = 0
    for i, img in enumerate(imgs):
        if img['status'] == 'success':
           arrays[counter] = get_target_object(img)
           counter = counter + 1
    return arrays

def encode_history(imgs, params, wtoi):
    max_length = ( params['max_length'] + 1 ) * params['max_rounds']
    N = get_success_num_qas(imgs)
    arrays = np.zeros((N, max_length), dtype='uint32')
    length = np.zeros(N, dtype='uint32')
    counter = 0
    for i,img in enumerate(imgs):
        if img['status'] == 'success':
            for j, qa in enumerate(img['qas']):
                length[counter] = min(max_length, len(qa['final_history']))
                for k,w in enumerate(qa['final_history']):
                    arrays[counter,k] = wtoi[w]
                counter += 1
    return arrays, length

def encode_question_qgen(imgs, params, wtoi):
    max_length = params['max_length']
    N = get_success_num_qas(imgs)
    arrays = np.zeros((N, max_length), dtype='uint32')
    length = np.zeros(N, dtype='uint32')
    img_id = np.zeros(N, dtype='uint32')
    counter = 0
    for i,img in enumerate(imgs):
        if img['status'] == 'success':
            for j, qa in enumerate(img['qas']):
                img_id[counter] = img['id']
                length[counter] = min(max_length, len(qa['final_question']))
                for k,w in enumerate(qa['final_question']):
                    arrays[counter,k] = wtoi[w]
                counter += 1
    return arrays, length, img_id

def right_align(seq, lengths):
    v = np.empty_like(seq)
    v[:] = 0
    N = seq.shape[1]
    for i in range(seq.shape[0]):
        v[i][N-lengths[i]: N] = seq[i][0: lengths[i]]
    return v

def getNonZeroLen(a):
    return (a != 0).sum(1)

def new_length(my_list):
    new_list = [x+1 for x in my_list]
    return new_list

def cut_ques_token(quesSucc_in, quesSucc_in_length):
    for i, a in enumerate(quesSucc_in_length):
        quesSucc_in[i][quesSucc_in_length[i]-1] = 0
    return quesSucc_in

def add_end_token(history, dial, quesSucc_in, quesSucc_out, quesSucc_id, params):
    assert(quesSucc_in.shape == quesSucc_out.shape)
    history_new = np.zeros((history.shape[0]+dial.shape[0], history.shape[1]))
    quesSucc_in_new = np.zeros((quesSucc_in.shape[0]+dial.shape[0], quesSucc_in.shape[1]))
    quesSucc_out_new = np.zeros((quesSucc_out.shape[0]+dial.shape[0], quesSucc_out.shape[1]))
    quesSucc_id_new = np.zeros((history.shape[0]+dial.shape[0]))
    history_length = getNonZeroLen(history)
    quesSucc_in_start = np.zeros(quesSucc_in.shape[1])
    quesSucc_in_start[0] = output_json['wtoi']['<START>']
    quesSucc_out_end = np.zeros(quesSucc_out.shape[1])
    quesSucc_out_end[0] = output_json['wtoi']['<END>']
    j = 0
    k = 0
    for i in xrange(history.shape[0]):
        if (history_length[i] == 0) and (i != 0):
            j = j + 1
        history_new[i+j] = history[i]
        quesSucc_in_new[i+j] = quesSucc_in[i]
        quesSucc_out_new[i+j] = quesSucc_out[i]
        quesSucc_id_new[i+j] = quesSucc_id[i]
    for i in xrange(dial.shape[0]+history.shape[0]):
        if not quesSucc_in_new[i].any():
            history_new[i] = dial[k]
            k = k + 1
            quesSucc_in_new[i] = quesSucc_in_start
            quesSucc_out_new[i] = quesSucc_out_end
            quesSucc_id_new[i] = quesSucc_id_new[i-1]
    return history_new, quesSucc_in_new, quesSucc_out_new, quesSucc_id_new
