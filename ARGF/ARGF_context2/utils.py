from torch.utils.data import Dataset
import pickle as pickle
import numpy as np
#import pickle
AUDIO = 'covarep'
VISUAL = 'facet'
TEXT = 'glove'
LABEL = 'label'
import numpy
# import sys
# import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('gb18030')
from data_prep import batch_iter, createOneHotMosei2way, get_raw_data


def total(params):
    '''
    count the total number of hyperparameter settings
    '''
    settings = 1
    for k, v in params.items():
        settings *= len(v)
    return settings


def load_pom(data_path):
    # parse the input args
    class POM(Dataset):
        '''
        PyTorch Dataset for POM, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    pom_data = pickle.load(open(data_path + "pom.pkl", 'rb'))
    pom_train, pom_valid, pom_test = pom_data['train'], pom_data['valid'], pom_data['test']

    train_audio, train_visual, train_text, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL], pom_train[TEXT], pom_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[TEXT], pom_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[TEXT], pom_test[LABEL]

    # code that instantiates the Dataset objects
    train_set = POM(train_audio, train_visual, train_text, train_labels)
    valid_set = POM(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = POM(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_iemocap(data_path, emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    iemocap_data = pickle.load(open(data_path + "iemocap.pkl", 'rb'))
    iemocap_train, iemocap_valid, iemocap_test = iemocap_data[emotion]['train'], iemocap_data[emotion]['valid'], \
                                                 iemocap_data[emotion]['test']
    #   print('iemocap_train',iemocap_train.shape)
    #   print('iemocap_valid',iemocap_valid.shape)
    #   print('iemocap_test',iemocap_test.shape)
    train_audio, train_visual, train_text, train_labels \
        = iemocap_train[AUDIO], iemocap_train[VISUAL], iemocap_train[TEXT], iemocap_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = iemocap_valid[AUDIO], iemocap_valid[VISUAL], iemocap_valid[TEXT], iemocap_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = iemocap_test[AUDIO], iemocap_test[VISUAL], iemocap_test[TEXT], iemocap_test[LABEL]
    print(train_audio.shape, 'audio_shape')
    print(train_labels.shape, 'labels_shape')
    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)

    print('iemocap_train', train_audio.shape)
    print('iemocap_valid', valid_audio.shape)
    print('iemocap_test', test_audio.shape)
    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_mosi(data_path):
    # parse the input args
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :], self.labels[idx]]

        def __len__(self):
            return self.audio.shape[0]

    #  mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'),encoding='bytes')
    mosi_data = pickle.load(open(data_path + "mosi.pkl", 'rb'))
    mosi_train, mosi_valid, mosi_test = mosi_data['train'], mosi_data['valid'], mosi_data['test']

    train_audio, train_visual, train_text, train_labels \
        = mosi_train[AUDIO], mosi_train[VISUAL], mosi_train[TEXT], mosi_train[LABEL]
    valid_audio, valid_visual, valid_text, valid_labels \
        = mosi_valid[AUDIO], mosi_valid[VISUAL], mosi_valid[TEXT], mosi_valid[LABEL]
    test_audio, test_visual, test_text, test_labels \
        = mosi_test[AUDIO], mosi_test[VISUAL], mosi_test[TEXT], mosi_test[LABEL]

    print(train_audio.shape)
    print(train_visual.shape)
    print(train_text.shape)
    print(train_labels.shape)

    train_audio2 = []
    train_visual2 = []
    train_text2 = []
    valid_audio2 = []
    valid_visual2 = []
    valid_text2 = []
    test_audio2 = []
    test_visual2 = []
    test_text2 = []
    train_labels2 = []
    valid_labels2 = []
    test_labels2 = []
    for i in range(train_audio.shape[0]):
        if numpy.any(numpy.isnan(train_audio)) or numpy.any(numpy.isnan(train_visual)) or numpy.any(
                numpy.isnan(train_text)):
            continue
        else:
            train_audio2.append(train_audio[i, :])
            train_visual2.append(train_visual[i, :])
            train_text2.append(train_text[i, :, :])
            train_labels2.append(train_labels[i, :])
    for i in range(valid_audio.shape[0]):
        if numpy.any(numpy.isnan(valid_audio)) or numpy.any(numpy.isnan(valid_visual)) or numpy.any(
                numpy.isnan(valid_text)):
            continue
        else:
            valid_audio2.append(valid_audio[i, :])
            valid_visual2.append(valid_visual[i, :])
            valid_text2.append(valid_text[i, :, :])
            valid_labels2.append(valid_labels[i, :])
    for i in range(test_audio.shape[0]):
        if numpy.any(numpy.isnan(test_audio)) or numpy.any(numpy.isnan(test_visual)) or numpy.any(
                numpy.isnan(test_text)):
            continue
        else:
            test_audio2.append(test_audio[i, :])
            test_visual2.append(test_visual[i, :])
            test_text2.append(test_text[i, :, :])
            test_labels2.append(test_labels[i, :])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_visual2 = np.array(train_visual2)
    print(train_visual2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_visual2 = np.array(valid_visual2)
    print(valid_visual2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_visual2 = np.array(test_visual2)
    test_text2 = np.array(test_text2)
    train_labels2 = np.array(train_labels2)
    valid_labels2 = np.array(valid_labels2)
    test_labels2 = np.array(test_labels2)

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio2, train_visual2, train_text2, train_labels2)
    valid_set = MOSI(valid_audio2, valid_visual2, valid_text2, valid_labels2)
    test_set = MOSI(test_audio2, test_visual2, test_text2, test_labels2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_mosei(data_path):
    # parse the input args
    class MOSEI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosei', 'text', 3), 'rb') as handle:
        u = pickle.Unpickler(handle)

        (train_text, train_label, _, _, test_text, test_label, _, train_length, _, test_length, _, _,
         _) = u.load()
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosei', 'audio', 3), 'rb') as handle:
        u = pickle.Unpickler(handle)

        (train_audio, train_label, _, _, test_audio, test_label, _, train_length, _, test_length, _, _,
         _) = u.load()
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosei', 'video', 3), 'rb') as handle:
        u = pickle.Unpickler(handle)

        (train_video, train_label, _, _, test_video, test_label, _, train_length, _, test_length, _, _,
         _) = u.load()

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(1800):
        for j in range(train_length[i]):
            train_audio2.append(train_audio[i][j][:])
            train_video2.append(train_video[i][j][:])
            train_text2.append(train_text[i][j][:])
            train_label2.append(train_label[i][j][:])
    for i in range(1800, 2250):
        for j in range(train_length[i]):
            valid_audio2.append(train_audio[i][j][:])
            valid_video2.append(train_video[i][j][:])
            valid_text2.append(train_text[i][j][:])
            valid_label2.append(train_label[i][j][:])
    for i in range(len(test_length)):
        for j in range(test_length[i]):
            test_audio2.append(test_audio[i][j][:])
            test_video2.append(test_video[i][j][:])
            test_text2.append(test_text[i][j][:])
            test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = MOSEI(train_audio2, train_video2, train_text2, train_label2)
    valid_set = MOSEI(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = MOSEI(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def createOneHot(train_label, test_label):
    print('train_label:', train_label)
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1],
                      maxlen + 1))  # [shape[0], shape[1], maxlen+1] batch size,length,classes
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in xrange(train_label.shape[0]):
        for j in xrange(train_label.shape[1]):
            train[i, j, int(train_label[i, j])] = 1

    for i in xrange(test_label.shape[0]):
        for j in xrange(test_label.shape[1]):
            test[i, j, int(test_label[i, j])] = 1

    return train, test


def load_mosi2(data_path):
    # parse the input args
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosi', 'text', 2), 'rb') as handle:
        u = pickle.Unpickler(handle)
        (train_text, train_label, test_text, test_label, maxlen, train_length, test_length) = u.load()
    #  (train_text, train_label, _, _, test_text, test_label, _, train_length, _, test_length, _, _, _) = u.load()
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosi', 'audio', 2), 'rb') as handle:
        u = pickle.Unpickler(handle)
        (train_audio, train_label, test_audio, test_label, maxlen, train_length, test_length) = u.load()
    #  (train_audio, train_label, _, _, test_audio, test_label, _, train_length, _, test_length, _, _, _) = u.load()
    with open('./dataset/{0}/raw/{1}_{2}way.pickle'.format('mosi', 'video', 2), 'rb') as handle:
        u = pickle.Unpickler(handle)
        (train_video, train_label, test_video, test_label, maxlen, train_length, test_length) = u.load()
    # (train_video, train_label, _, _, test_video, test_label, _, train_length, _, test_length, _, _,_) = u.load()

    print(train_audio.shape)
    print(train_video.shape)
    print(train_text.shape)
    print(train_label.shape)

    test = []
    for i in range(len(test_length)):
        for j in range(test_length[i]):
            test.append(test_label[i][j])
    test = np.array(test)
    np.save('label.npy', test)

    train_label, test_label = createOneHot(train_label, test_label)

    #   train_label = train_label[:,:,np.newaxis]
    #   test_label = test_label[:,:,np.newaxis]
    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(49):
        for j in range(train_length[i]):
            train_audio2.append(train_audio[i][j][:])
            train_video2.append(train_video[i][j][:])
            train_text2.append(train_text[i][j][:])
            train_label2.append(train_label[i][j][:])
    for i in range(49, 62):
        for j in range(train_length[i]):
            valid_audio2.append(train_audio[i][j][:])
            valid_video2.append(train_video[i][j][:])
            valid_text2.append(train_text[i][j][:])
            valid_label2.append(train_label[i][j][:])
    for i in range(len(test_length)):
        for j in range(test_length[i]):
            test_audio2.append(test_audio[i][j][:])
            test_video2.append(test_video[i][j][:])
            test_text2.append(test_text[i][j][:])
            test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = MOSI(train_audio2, train_video2, train_text2, train_label2)
    valid_set = MOSI(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = MOSI(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_iemocap6(data_path):
    # parse the input args
    class iemocap6(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    data = 'iemocap'
    classes = 6
    train_data, test_data, train_audio, test_audio, train_text, test_text, train_video, test_video, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data( \
        data, classes)
    print(train_label.shape, 'train_label')

    '''
    with open('unimodal_{0}_{1}way.pickle'.format('mosi', 2), 'rb') as handle:
            u = pickle.load(handle)

    train_mask=u['train_mask']
    test_mask=u['test_mask']
    train_label=u['train_label']
    test_label=u['test_label']
        
    train_text = u['text_train']
    train_audio = u['audio_train']
    train_video = u['video_train']

    test_text = u['text_test']
    test_audio = u['audio_test']
    test_video = u['video_test']
    '''

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(96):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                train_audio2.append(train_audio[i][j][:])
                train_video2.append(train_video[i][j][:])
                train_text2.append(train_text[i][j][:])
                train_label2.append(train_label[i][j][:])
    for i in range(96, 120):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                valid_audio2.append(train_audio[i][j][:])
                valid_video2.append(train_video[i][j][:])
                valid_text2.append(train_text[i][j][:])
                valid_label2.append(train_label[i][j][:])
    for i in range(test_text.shape[0]):
        for j in range(test_audio.shape[1]):
            if test_mask[i, j] == 1:
                test_audio2.append(test_audio[i][j][:])
                test_video2.append(test_video[i][j][:])
                test_text2.append(test_text[i][j][:])
                test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = iemocap6(train_audio2, train_video2, train_text2, train_label2)
    valid_set = iemocap6(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = iemocap6(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_iemocap6_context(data_path):
    # parse the input args
    class iemocap6(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    data = 'iemocap'
    classes = 6
    #   train_data, test_data, train_audio, test_audio, train_text, test_text, train_video, test_video, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(\
    #           data, classes)
    #   print(train_label.shape,'train_label')

    with open('unimodal_{0}_{1}way.pickle'.format('iemocap', 6), 'rb') as handle:
        u = pickle.load(handle)

    train_mask = u['train_mask']
    test_mask = u['test_mask']
    train_label = u['train_label']
    test_label = u['test_label']

    train_text = u['text_train']
    train_audio = u['audio_train']
    train_video = u['video_train']

    test_text = u['text_test']
    test_audio = u['audio_test']
    test_video = u['video_test']

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(96):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                train_audio2.append(train_audio[i][j][:])
                train_video2.append(train_video[i][j][:])
                train_text2.append(train_text[i][j][:])
                train_label2.append(train_label[i][j][:])
    for i in range(96, 120):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                valid_audio2.append(train_audio[i][j][:])
                valid_video2.append(train_video[i][j][:])
                valid_text2.append(train_text[i][j][:])
                valid_label2.append(train_label[i][j][:])
    for i in range(test_text.shape[0]):
        for j in range(test_audio.shape[1]):
            if test_mask[i, j] == 1:
                test_audio2.append(test_audio[i][j][:])
                test_video2.append(test_video[i][j][:])
                test_text2.append(test_text[i][j][:])
                test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = iemocap6(train_audio2, train_video2, train_text2, train_label2)
    valid_set = iemocap6(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = iemocap6(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_iemocap4(data_path):
    # parse the input args
    class iemocap4(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    with open('unimodal_{0}_{1}way.pickle'.format('iemocap', 4), 'rb') as handle:
        u = pickle.load(handle)

    train_mask = u['train_mask']
    test_mask = u['test_mask']
    train_label = u['train_label']
    test_label = u['test_label']

    train_text = u['text_train']
    train_audio = u['audio_train']
    train_video = u['video_train']

    test_text = u['text_test']
    test_audio = u['audio_test']
    test_video = u['video_test']

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(96):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                train_audio2.append(train_audio[i][j][:])
                train_video2.append(train_video[i][j][:])
                train_text2.append(train_text[i][j][:])
                train_label2.append(train_label[i][j][:])
    for i in range(96, 120):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                valid_audio2.append(train_audio[i][j][:])
                valid_video2.append(train_video[i][j][:])
                valid_text2.append(train_text[i][j][:])
                valid_label2.append(train_label[i][j][:])
    for i in range(test_text.shape[0]):
        for j in range(test_audio.shape[1]):
            if test_mask[i, j] == 1:
                test_audio2.append(test_audio[i][j][:])
                test_video2.append(test_video[i][j][:])
                test_text2.append(test_text[i][j][:])
                test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = iemocap4(train_audio2, train_video2, train_text2, train_label2)
    valid_set = iemocap4(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = iemocap4(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_mosi_context(data_path):
    # parse the input args
    class mosi_context(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    with open('unimodal_{0}_{1}way.pickle'.format('mosi', 2), 'rb') as handle:
        u = pickle.load(handle)
  #  data_path = 'unimodal_{0}_{1}way.pickle'.format('mosi', 2)
  #  u = pickle.load(open(data_path, 'rb'))
    train_mask = u['train_mask']
    test_mask = u['test_mask']
    train_label = u['train_label']
    test_label = u['test_label']

    train_text = u['text_train']
    train_audio = u['audio_train']
    train_video = u['video_train']

    test_text = u['text_test']
    test_audio = u['audio_test']
    test_video = u['video_test']

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(49):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                train_audio2.append(train_audio[i][j][:])
                train_video2.append(train_video[i][j][:])
                train_text2.append(train_text[i][j][:])
                train_label2.append(train_label[i][j][:])
    for i in range(49, 62):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                valid_audio2.append(train_audio[i][j][:])
                valid_video2.append(train_video[i][j][:])
                valid_text2.append(train_text[i][j][:])
                valid_label2.append(train_label[i][j][:])
    for i in range(test_text.shape[0]):
        for j in range(test_audio.shape[1]):
            if test_mask[i, j] == 1:
                test_audio2.append(test_audio[i][j][:])
                test_video2.append(test_video[i][j][:])
                test_text2.append(test_text[i][j][:])
                test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = mosi_context(train_audio2, train_video2, train_text2, train_label2)
    valid_set = mosi_context(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = mosi_context(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_mosei_context(data_path):
    # parse the input args
    class mosei_context(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :], self.labels[idx]]

        #        return [self.audio, self.visual, self.text, self.labels]
        def __len__(self):
            return self.audio.shape[0]

    with open('unimodal_{0}_{1}way.pickle'.format('mosei', 3), 'rb') as handle:
        u = pickle.load(handle)

    train_mask = u['train_mask']
    test_mask = u['test_mask']
    train_label = u['train_label']
    test_label = u['test_label']

    train_text = u['text_train']
    train_audio = u['audio_train']
    train_video = u['video_train']

    test_text = u['text_test']
    test_audio = u['audio_test']
    test_video = u['video_test']

    train_audio2 = []
    train_video2 = []
    train_text2 = []
    valid_audio2 = []
    valid_video2 = []
    valid_text2 = []
    test_audio2 = []
    test_video2 = []
    test_text2 = []
    train_label2 = []
    valid_label2 = []
    test_label2 = []
    for i in range(1800):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                train_audio2.append(train_audio[i][j][:])
                train_video2.append(train_video[i][j][:])
                train_text2.append(train_text[i][j][:])
                train_label2.append(train_label[i][j][:])
    for i in range(1800, 2250):
        for j in range(train_audio.shape[1]):
            if train_mask[i, j] == 1:
                valid_audio2.append(train_audio[i][j][:])
                valid_video2.append(train_video[i][j][:])
                valid_text2.append(train_text[i][j][:])
                valid_label2.append(train_label[i][j][:])
    for i in range(test_text.shape[0]):
        for j in range(test_audio.shape[1]):
            if test_mask[i, j] == 1:
                test_audio2.append(test_audio[i][j][:])
                test_video2.append(test_video[i][j][:])
                test_text2.append(test_text[i][j][:])
                test_label2.append(test_label[i][j][:])

    # code that instantiates the Dataset objects
    train_audio2 = np.array(train_audio2)
    print(train_audio2.shape, 'train_audio2')
    train_video2 = np.array(train_video2)
    print(train_video2.shape, 'train_video2')
    train_text2 = np.array(train_text2)
    print(train_text2.shape, 'train_text2')
    valid_audio2 = np.array(valid_audio2)
    print(valid_audio2.shape, 'valid_audio2')
    valid_video2 = np.array(valid_video2)
    print(valid_video2.shape, 'valid_video2')
    valid_text2 = np.array(valid_text2)
    test_audio2 = np.array(test_audio2)
    test_video2 = np.array(test_video2)
    test_text2 = np.array(test_text2)
    train_label2 = np.array(train_label2)
    valid_label2 = np.array(valid_label2)
    test_label2 = np.array(test_label2)
    train_set = mosei_context(train_audio2, train_video2, train_text2, train_label2)
    valid_set = mosei_context(valid_audio2, valid_video2, valid_text2, valid_label2)
    test_set = mosei_context(test_audio2, test_video2, test_text2, test_label2)

    audio_dim = train_set[0][0].shape[0]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[0]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[0]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims
