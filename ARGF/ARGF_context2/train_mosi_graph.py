from __future__ import print_function
# from model import LMF
from utils import total, load_mosi2, load_mosi_context
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from model import *
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import itertools
#from thop import profile


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print(cuda, 'cuda')


def display(bi_acc, f1):
    print("Binary accuracy on test set is {}".format(bi_acc))
    print("F1-score on test set is {}".format(f1))


def main(options):
    DTYPE = torch.FloatTensor

    # parse the input args
    run_id = options['run_id']
    epochs = options['epochs']
    data_path = options['data_path']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    output_dim = options['output_dim']

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, run_id))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, run_id))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))

    train_set, valid_set, test_set, input_dims = load_mosi_context(data_path)

    params = dict()

    params['audio_hidden'] = [150, 50, 100]
    params['audio_dropout'] = [0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    #   params['video_dropout'] = [0, 0.1,  0.2, 0.3, 0.5,0.6,0.7,0.8]
    #    params['text_dropout'] = [0, 0.1,  0.2, 0.3, 0.5,0.6,0.7,0.8]
    params['learning_rate'] = [0.0001, 0.001, 0.01]
    params['batch_size'] = [8, 16, 32]
    params['weight_decay'] = [0, 0.001, 0.01, 0.0001]
    params['alpha'] = [0.01, 0.001, 0.05]

    # params['alpha']=[0.05]
    total_settings = total(params)
    print("There are {} different hyper-parameter settings in total.".format(total_settings))
    audio_in = input_dims[0]
    video_in = input_dims[1]
    text_in = input_dims[2]
    seen_settings = set()

    with open(output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(
            ["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
             'learning_rate', 'batch_size', 'weight_decay',
              'Test binary accuracy', 'Test f1_score', 'alpha'])
    best_acc = 0
    best_f1 = 0
    kk = 0
    #  for i in range(total_settings):
    for i in range(1000000):
        if kk >= total_settings:
            break
        
        ahid = random.choice(params['audio_hidden'])
        vhid = ahid
        thid = ahid
        thid_2 = thid 
        adr = random.choice(params['audio_dropout'])
        vdr = adr
        tdr = adr
        lr = random.choice(params['learning_rate'])
        batch_sz = random.choice(params['batch_size'])
        decay = random.choice(params['weight_decay'])
        alpha = random.choice(params['alpha'])
        
        '''
        ahid = 50  #######50
        vhid = ahid
        thid = ahid
        thid_2 = thid
        adr = 0.5  ###########0.6
        vdr = adr
        tdr = adr
        lr = 0.001  ########0.001
        batch_sz = 16  ############32,16
        decay = 0.005  #########
        alpha = 0.001  ###0.05
        '''
        # reject the setting if it has been tried
        current_setting = (ahid, vhid, thid, adr, vdr, tdr, lr, batch_sz, decay, alpha)
        if i == 0:
            best_setting = current_setting
        if (current_setting in seen_settings) and (current_setting != best_setting):
            continue
        else:
            seen_settings.add(current_setting)
            kk += 1
        latent_dim = 10

        encoder_a = Encoder_5(audio_in, ahid, adr).cuda()  #####zhe ge ke yi de
        encoder_v = Encoder_5(video_in, vhid, vdr).cuda()
        encoder_l = Encoder_5(text_in, thid, dropout=tdr).cuda()

        decoder_a = Decoder2(ahid, audio_in).cuda()
        decoder_v = Decoder2(vhid, video_in).cuda()
        decoder_l = Decoder2(thid, text_in).cuda()
        discriminator = Discriminator(ahid).cuda()
        #     classifier = Classifier(hidden=thid,input_dim=output_dim,layer_size=64).cuda()
        classifier = graph11_new(in_size=thid, output_dim=output_dim).cuda()
        classifier_3 = classifier3(in_size=thid, output_dim=output_dim).cuda()
        b1 = 0.5
        b2 = 0.999


        #  criterion = nn.L1Loss(size_average=False).cuda()
        criterion = nn.L1Loss(size_average=False).cuda()
        adversarial_loss = torch.nn.BCELoss().cuda()
        classifier_loss = torch.nn.SoftMarginLoss().cuda()
        pixelwise_loss = torch.nn.L1Loss(size_average=False).cuda()
        #     pixelwise_loss = torch.nn.MSELoss().cuda()
        #     pixelwise_loss = torch.nn.KLDivLoss().cuda()    #bad

        optimizer_G = torch.optim.Adam(
            itertools.chain(encoder_a.parameters(), encoder_v.parameters(), encoder_l.parameters(), \
                            decoder_a.parameters(), decoder_l.parameters(), decoder_v.parameters()), weight_decay=decay,
            lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
        optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(b1, b2), weight_decay=decay)
        optimizer_E = torch.optim.Adam(
            itertools.chain(encoder_a.parameters(), encoder_v.parameters(), encoder_l.parameters(),
                            classifier_3.parameters()), lr=lr, betas=(b1, b2), weight_decay=decay)
        '''
        optimizer_G = torch.optim.Adam(itertools.chain(encoder_a.parameters(), encoder_v.parameters(),encoder_l.parameters(),\
                                decoder_a.parameters(),decoder_l.parameters(),decoder_v.parameters()), weight_decay=decay, lr=lr)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=decay)
        optimizer_C = torch.optim.Adam(itertools.chain(encoder_a.parameters(), encoder_v.parameters(),encoder_l.parameters(),classifier.parameters()), lr=lr, weight_decay=decay)
        optimizer_E = torch.optim.Adam(itertools.chain(encoder_a.parameters(), encoder_v.parameters(),encoder_l.parameters(),classifier_3.parameters()), lr=lr, weight_decay=decay)
        '''

        if cuda:
            DTYPE = torch.cuda.FloatTensor

        # setup training
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
        valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
        test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
        curr_patience = patience

        for e in range(epochs):

            avg_train_loss = 0.0
            avg_closs = 0.0
            avg_rloss = 0.0
            avg_v = 0.0
            avg_a = 0.0
            avg_l = 0.0
            for batch in train_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                # encoder-decoder
                optimizer_G.zero_grad()
                a_en = encoder_a(x_a)
                v_en = encoder_v(x_v)
                l_en = encoder_l(x_t)
                #    z = Variable(Tensor(np.random.normal(0, 1, (y.size(0), latent_dim))))
                #     ade = torch.cat([a_en,z],1)
                #    vde = torch.cat([v_en,z],1)
                #    lde = torch.cat([l_en,z],1)
                a_de = decoder_a(a_en)
                v_de = decoder_v(v_en)
                l_de = decoder_l(l_en)

                rl1 = pixelwise_loss(a_de, x_a) + pixelwise_loss(v_de, x_v) + pixelwise_loss(l_de, x_t)   #####reconstruction loss
                avg_rloss += rl1.item() / len(train_set)
                valid = Variable(torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.cuda.FloatTensor(y.shape[0], 1).fill_(0.0), requires_grad=False)

                g_loss = alpha * (adversarial_loss(discriminator(l_en), valid) + adversarial_loss(discriminator(v_en),
                                                                                                  valid)) + (1 - alpha) * (rl1)
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

                # classifier
                optimizer_E.zero_grad()
                #  v_a = torch.cat([a_en,v_en],1)
                #   fusion = torch.cat([v_a,l_en],1)
                #   v_c = classifier_3(fusion)
                a = classifier_3(a_en)
                v = classifier_3(v_en)
                l = classifier_3(l_en)
                c_loss = criterion(a, y) + criterion(l, y) + criterion(v, y)     ######classification loss
                c_loss.backward(retain_graph=True)
                optimizer_E.step()
                avg_closs += c_loss.item() / len(train_set)
                # discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(a_en), valid)
                avg_v += torch.sum(discriminator(v_en).squeeze().data) / (len(train_set) * batch_sz)

                fake_loss = adversarial_loss(discriminator(v_en), fake) + adversarial_loss(discriminator(l_en), fake)
                avg_a += torch.sum(discriminator(a_en).squeeze().data) / (len(train_set) * batch_sz)
                avg_l += torch.sum(discriminator(l_en).squeeze().data) / (len(train_set) * batch_sz)
                d_loss = 0.5 * (real_loss + fake_loss)    ##### discrimination loss for discriminator
                #    print(d_loss,'d_loss')
                d_loss.backward(retain_graph=True)
                optimizer_D.step()

                # fusion
                v_a = torch.cat([a_en.unsqueeze(1), v_en.unsqueeze(1)], 1)
                fusion = torch.cat([v_a, l_en.unsqueeze(1)], 1)

                optimizer_C.zero_grad()
                output, _ = classifier(fusion)
                loss = criterion(output, y)
                loss.backward()
                optimizer_C.step()
                avg_loss = loss.item()
                avg_train_loss += avg_loss / len(train_set)

            print(
                "Epoch: {} loss: {}".format(e, avg_train_loss))

            # Terminate the training process if run into NaN
            if np.isnan(avg_train_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            #  model.eval()
            avg_valid_loss = 0
            #   k = 0
            for batch in valid_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                a_en = encoder_a(x_a)
                v_en = encoder_v(x_v)
                l_en = encoder_l(x_t)
                '''
                v_a = torch.cat([a_en.unsqueeze(2),v_en.unsqueeze(2)],2)
                fusion = torch.cat([v_a,l_en.unsqueeze(2)],2)
                fusion = fusion.unsqueeze(1)
                '''
                v_a = torch.cat([a_en.unsqueeze(1), v_en.unsqueeze(1)], 1)
                fusion = torch.cat([v_a, l_en.unsqueeze(1)], 1)

                output, _ = classifier(fusion)
                valid_loss = criterion(output, y)
                avg_valid_loss += valid_loss.item()
            #    k+=1
            y = y.cpu().data.numpy().reshape(-1, output_dim)
            #  print(k,'k')
            if np.isnan(avg_valid_loss):
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / len(valid_set)
            print("Validation loss is: {}".format(avg_valid_loss))

            if (avg_valid_loss < min_valid_loss):
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                torch.save(encoder_a, './models/encoder_a_{}.pkl'.format(run_id))
                torch.save(encoder_v, './models/encoder_v_{}.pkl'.format(run_id))
                torch.save(encoder_l, './models/encoder_l_{}.pkl'.format(run_id))
                torch.save(classifier, './models/classifier_{}.pkl'.format(run_id))
                print("Found new best model, saving to disk...")
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break
            print("\n\n")

            avg_test_loss = 0
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                #      output_test = model(x_a, x_v, x_t)
                a_en = encoder_a(x_a)
                v_en = encoder_v(x_v)
                l_en = encoder_l(x_t)
                '''
                v_a = torch.cat([a_en.unsqueeze(2),v_en.unsqueeze(2)],2)
                fusion = torch.cat([v_a,l_en.unsqueeze(2)],2)
                fusion = fusion.unsqueeze(1)
                '''
                v_a = torch.cat([a_en.unsqueeze(1), v_en.unsqueeze(1)], 1)
                fusion = torch.cat([v_a, l_en.unsqueeze(1)], 1)
                output_test, _ = classifier(fusion)
                loss_test = criterion(output_test, y)
                avg_test_loss += loss_test.item() / len(test_set)

            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)

            # these are the needed metrics
            output_test = output_test.reshape((len(output_test) * output_dim,))
            y = y.reshape((len(y) * output_dim,))
            mae = np.mean(np.absolute(output_test - y))
        k = 0
        true_acc = 0
        true_f1 = 0
        if complete:

            #   best_model = torch.load(model_path)
            #   best_model.eval()
            encoder_a = torch.load('./models/encoder_a_{}.pkl'.format(run_id))
            encoder_v = torch.load('./models/encoder_v_{}.pkl'.format(run_id))
            encoder_l = torch.load('./models/encoder_l_{}.pkl'.format(run_id))
            classifier = torch.load('./models/classifier_{}.pkl'.format(run_id))
            k = 0
            for batch in test_iterator:
                x = batch[:-1]
                x_a = Variable(x[0].float().type(DTYPE), requires_grad=False).squeeze()
                x_v = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze()
                x_t = Variable(x[2].float().type(DTYPE), requires_grad=False)
                y = Variable(batch[-1].view(-1, output_dim).float().type(DTYPE), requires_grad=False)
                a_en = encoder_a(x_a)
                v_en = encoder_v(x_v)
                l_en = encoder_l(x_t)
                '''
                v_a = torch.cat([a_en.unsqueeze(2),v_en.unsqueeze(2)],2)
                fusion = torch.cat([v_a,l_en.unsqueeze(2)],2)
                fusion = fusion.unsqueeze(1)
                '''
                v_a = torch.cat([a_en.unsqueeze(1), v_en.unsqueeze(1)], 1)
                fusion = torch.cat([v_a, l_en.unsqueeze(1)], 1)
                output_test, weights = classifier(fusion)
                loss_test = criterion(output_test, y)
                '''
                true_label = (y >= 0)
                predicted_label = (output_test >= 0)
                if k == 0 :
                   total_y = y
                   total_test = output_test
                else:
                   total_y = torch.cat([total_y,y],0)
                   total_test = torch.cat([total_test,output_test],0)
                k += 1
                '''
            output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
            y = y.cpu().data.numpy().reshape(-1, output_dim)
            print(output_test.shape, 'output_test_shape')
            print(y.shape, 'y_shape')

            output_test2 = output_test.reshape((len(output_test) * output_dim,))
            y2 = y.reshape((len(y) * output_dim,))

            true_label = np.argmax(y, 1)
            predicted_label = np.argmax(output_test, 1)
            bi_acc = accuracy_score(true_label, predicted_label)
            f1 = f1_score(true_label, predicted_label, average='weighted')
            display(bi_acc, f1)

            if bi_acc > best_acc:
                best_acc = bi_acc
                best_setting = current_setting
                '''
                np.save('fusion.npy',fusion.cpu().data.numpy())
                np.save('label_one_hot.npy',y)
                np.save('weights.npy',weights.cpu().data.numpy())
                np.save('predicted.npy',output_test)
                '''
            if f1 > best_f1:
                best_f1 = f1
            print('best_acc: ', best_acc)
            print('best_f1: ', best_f1)
            print('best_setting: ', best_setting)
            '''
            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([ahid, vhid, thid, adr, vdr, tdr,  lr, batch_sz, decay,
                                 bi_acc, f1,alpha])
            '''


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=240)
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='mosi')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='./data/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='result2')
    OPTIONS.add_argument('--max_len', dest='max_len', type=int, default=20)
    PARAMS = vars(OPTIONS.parse_args())

    main(PARAMS)
