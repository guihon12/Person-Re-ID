import numpy as np
import os
import cv2
import random



def get_pair(path, set, num_id, positive):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break
    for i in range(2):
        filepath = ''
        while True:
            index = int(random.random() * 10)
            filepath = '%s/%s/%04d_%02d.jpg' % (path, set, id[i], index)
            if not os.path.exists(filepath):
                continue
            break
        pair.append(filepath)
    return pair


def get_num_id(path, set):
    files = os.listdir('%s/%s' % (path, set))
    files.sort()
    return int(files[-1].split('_')[0]) - int(files[0].split('_')[0]) + 1


def read_data(path, set, num_id, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    for i in range(batch_size // 2):
        pairs = [get_pair(path, set, num_id, True), get_pair(path, set, num_id, False)]
        for pair in pairs:
            images = []
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
        labels.append([1., 0.])
        labels.append([0., 1.])
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


def gallery_make(path, set, num_id, state):
    gallery = []
    while True:
        index = int(random.random() * 10)
        filepath = '%s/%s/%04d_%02d.jpg' % (path, set, state, index)
        if not os.path.exists(filepath):
            continue
        if index != 0:
            break
    gallery.append(filepath)
    for i in range(99):
        while True:
            id = int(random.random()*num_id)
            index = int(random.random() * 10)
            filepath = '%s/%s/%04d_%02d.jpg' % (path, set, id, index)
            if not os.path.exists(filepath):
                continue
            if id != state:
                break
        gallery.append(filepath)
    return gallery


def single_data(path, set, num_id):
    query = []
    gallery = []
    for i in range(num_id):
        filepath = '%s/%s/%04d_00.jpg' % (path, set, i)
        if not os.path.exists(filepath):
            print('no filepath!!')
            break
        query.append(filepath)
        tmp = gallery_make(path, set, num_id, i)
        gallery.append(tmp)
    return query, gallery


def gallery_make2(path, set, query, state):
    file_path = '%s/%s/' % (path, set)
    gallery = os.listdir(file_path)
    random.shuffle(gallery)
    out = []
    tmp = []
    count = 0
    for image in gallery:
        if query in '%s/%s/%s' %(path, set, image):
            continue
        elif '%04d' % state in image:
            if count == 5:
                continue
            out.append(image)
            count = count+1
        else:
            tmp.append(image)
        gallery[gallery.index(image)] = '%s/%s/' % (path, set) + image
    print(len(tmp))
    for i in range(len(tmp)):
        out.append(tmp[i])
    for image in out:
        out[out.index(image)] = '%s/%s/' % (path, set) + image
    return out


def multi_data(path, set, num_id):
    query = []
    file_path = '%s/%s/' % (path, set)
    gallery = os.listdir(file_path)
    for image in gallery:
        gallery[gallery.index(image)] = '%s/%s/' % (path, set) + image
    for i in range(num_id):
        while True:
            id = int(random.random() * 10)
            filepath = '%s/%s/%04d_%02d.jpg' % (path, set, i, id)
            if not os.path.exists(filepath):
                continue
            break
        query.append(filepath)
    return query, gallery


def pairing(query, gallery):
    pair = []
    pair.append(query)
    pair.append(gallery)
    pair.append(gallery)
    return pair


def multi_batch_data(query, gallery, state, image_width, image_height):
    #for g in gallery:
    #   if query[state] == g:
    #        tmp = g
    #gallery.remove(tmp)
    batch_images = []
    labels = []
    for i in range(len(gallery)):
        pairs = []
        pairs.append(pairing(query[state], gallery[i]))
        if '%04d' % state in gallery[i]:
            labels.append([1., 0., 0., 0.])
        else:
            labels.append([0., 0., 0., 1.])
        images = []
        for pair in pairs:
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)
    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)


def single_batch_data(query, gallery, image_width, image_height, state, batch_size):
    batch_images = []
    labels = []
    for j in range(batch_size):
        pairs = []
        pairs.append(pairing(query[state], gallery[state][j]))
        if j == 0:
            labels.append([1., 0., 0., 0.])
        else:
            labels.append([0., 0., 0., 1.])
        images = []
        for pair in pairs:
            for p in pair:
                image = cv2.imread(p)
                image = cv2.resize(image, (image_width, image_height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
            batch_images.append(images)

    return np.transpose(batch_images, (1, 0, 2, 3, 4)), np.array(labels)
