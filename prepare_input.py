
import json
import os
import pandas as pd

from collections import Counter


def load_pid(csv_file):
    df = pd.read_csv(csv_file)
    return df['pid'].tolist()

def load_data(csv_file):
    df = pd.read_csv(csv_file, header=None)
    return df.values


test_pid_csv = 'test_pid_shanghai_balanced_resnet50_fea_0.csv'
train_pid_csv = 'train_pid_shanghai_balanced_resnet50_fea_0.csv'
test_data_csv = 'test_shanghai_balanced_resnet50_fea_0.csv'
train_data_csv = 'train_shanghai_balanced_resnet50_fea_0.csv'

test_pid = load_pid(os.path.join('/mnt/Data/report', test_pid_csv))
train_pid = load_pid(os.path.join('/mnt/Data/report', train_pid_csv))

test_feat = load_data(os.path.join('/mnt/Data/report', test_data_csv))
train_feat = load_data(os.path.join('/mnt/Data/report', train_data_csv))

pids = test_pid + train_pid
print('test pid:', len(test_pid), ' train pid:', len(train_pid))
print('total pids:', len(pids), len(set(pids)))
# print(test_pid)

print(type(test_feat), test_feat.shape)
print(type(train_feat), train_feat.shape)
# print(test_data)

id_feat_map = dict()
for i in range(len(test_pid)):
    id_feat_map[test_pid[i]] = test_feat[i]
for i in range(len(train_pid)):
    id_feat_map[train_pid[i]] = train_feat[i]


data_dir = '/mnt/Data/report/text_report'
json_list = [f for f in os.listdir(data_dir) if f.endswith('.json')]
print('total json files:', len(json_list))

id_map_jsons = dict()
for jf in json_list:
    if jf.startswith('IMG'):
        pid = jf[:-5].split('_')[-1]
    else:
        pid = jf[:-5].split(' ')[0]
        pid = pid.split('(')[0]
    pid = int(pid)
    if pid not in id_map_jsons.keys():
        id_map_jsons[pid] = []
    id_map_jsons[pid].append(jf)

print('patient id:', len(id_map_jsons.keys()))
print(id_map_jsons[103455])




key = '镜下所见'

id_text = dict()
word_cnt = Counter()

for pid in pids:
    if pid not in id_map_jsons.keys():
        #print('no pid json:', pid)
        continue
    
    for js in id_map_jsons[pid]:
        json_file = os.path.join(data_dir, js)
        with open(json_file, 'r') as f:
            report = json.load(f)
        if len(report[key]) == 0:
            #print('key length zero:', js)
            continue
        id_text[pid] = report[key]
        word_cnt.update(report[key].split('/'))

max_len_caption = 0
for pid, rp in id_text.items():
    max_len_caption = max(max_len_caption, len(rp.split('/')))
    #if 103 == len(rp.split('/')):
        #print(pid, rp.split('/'))
        #words = rp.split('/')
        #for k in range(len(words)):
        #    print(k, words[k])

print('max_len_caption:', max_len_caption)


# print(word_cnt)
# print(id_text)
words = [w for w in word_cnt if word_cnt[w] > 1]
words += ['<unk>', '<start>', '<end>', '<pad>']
# print(words)


word_index = dict()
word_index['word_to_index'] = dict()
word_index['index_to_word'] = dict()

for idx in range(len(words)):
    word_index['index_to_word'][int(idx)] = words[idx]
    word_index['word_to_index'][str(words[idx])] = idx

with open('/mnt/Data/report/word_index.json', 'w', encoding='utf-8') as f:
    json.dump(word_index, f, sort_keys=True, indent=4, ensure_ascii=False)



# train_data
def prepare_data(pids, save_filename, id_text, word_to_index, id_feat_map, max_len_caption):
    data = dict()
    for pid in pids:
        if pid not in id_text.keys():
            #print('not found key:', pid)
            continue
        report = id_text[pid]

        record = dict()
        record['text'] = report
        vec = [word_to_index['<start>']]
        for w in report.split('/'):
            if w in word_to_index.keys():
                vec.append(word_to_index[w])
            else:
                vec.append(word_to_index['<unk>'])
        vec.append(word_to_index['<end>'])
        record['cap_length'] = len(vec)
        if len(vec) < max_len_caption + 2:
            vec += [word_to_index['<pad>']] * (max_len_caption + 2 - len(vec))

        record['vector'] = vec
        record['feature'] = id_feat_map[pid].tolist()
        data[pid] = record

    print('valid train/test data:', len(data.keys()))
    save_filename = os.path.join('/mnt/Data/report', save_filename)
    with open(save_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4, ensure_ascii=False)


prepare_data(train_pid, 'train_data.json', id_text, word_index['word_to_index'], id_feat_map, max_len_caption)

prepare_data(test_pid, 'test_data.json', id_text, word_index['word_to_index'], id_feat_map, max_len_caption)

