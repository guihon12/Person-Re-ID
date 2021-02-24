import numpy as np
import math
import os

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff))
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist

# npy파일 경로
npy_dir = '../data/npy/val'

# test 데이터 ID들
TEST_imgls = [s1 for s1 in os.listdir(npy_dir) if not ".txt" in s1]

all_img_name = []
cnt_img = []
start_num = [0]

# cnt_img : id별 이미지 수
# all_img_name : 모든 이미지
for name in TEST_imgls:
    name_dir = npy_dir + '/' + name
    cnt_img.append(len(os.listdir(name_dir)))
    all_img_name.extend([name_dir + '/' + s2.split('.')[0] for s2 in os.listdir(name_dir) if not ".txt" in s2])

for u in range(len(cnt_img)):
    tmp = 0
    for v in range(u+1):
        tmp += cnt_img[v]
    start_num.append(tmp)

cnt_T = [0 for i in range(20)]
hw_cnt = 0
emb = []
mAP = 0.0

for n in all_img_name:
    npy_ = n +'.npy'
    load_emb = np.load(npy_)
    emb.append(load_emb)

for x in range(len(all_img_name)):
    euclidean_distance = []
    rank_img = []
    rank_img_index = []
    for y in range(len(all_img_name)):
        euclidean_distance.append(distance(emb[x], emb[y], 0))
    sorted_euclidean_distance = sorted(euclidean_distance)
    for i in range(0, 21):
        rank_img_index.append(euclidean_distance.index(sorted_euclidean_distance[i]))
        rank_img.append(all_img_name[rank_img_index[i]])
    rank_cnt = [0 for i in range(21)]
    print('-------------------')
    print(all_img_name[x])
    print(rank_img[1])
    if all_img_name[x].split('_')[1] == rank_img[1].split('_')[1]:
        hw_cnt += 1
    print('rank1: %f' % float((hw_cnt)/(x+1)))
    print('-------------------')
    for j in range(1, 21):
        if all_img_name[x].split('_')[1] == rank_img[j].split('_')[1]:
            rank_cnt[j] = 1
    AP = 0
    tmp = 0
    for i in range(1,21):
        if rank_cnt[i] == 1:
            tmp += 1
            AP += tmp / i
    if tmp == 0:
        AP = 0
    else:
        AP = AP / tmp
    mAP += AP

    for i in range(1, 21):
        for j in range(i):
            if rank_cnt[j + 1] == 1:
                cnt_T[i - 1] += 1
                break

mAP = mAP / len(all_img_name)
print('mAP :', mAP)

for i in range(1, 21):
    #print('Rank', i)
    print(cnt_T[i - 1] / len(all_img_name))

# acc_txt.write('\n accuracy : %f     %d %d' % (Rank1_accuracy, cnt_T, cnt_F))
