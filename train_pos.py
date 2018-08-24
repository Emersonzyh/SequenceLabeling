from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from model import labelingModel
from DataSet_pos import DataSet
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class LBconfig():
    single_word_len = 10  # fro cnn
    vocab_size = 0
    n_input = 100 # word2vec data size
    char_num = 166
    n_hidden = 100 # hidden layer num of features
    num_layers = 1 # number of lstm layers
    epsilon=1e-4
    tag_size = 29 # TAGs total classes (86 types)

    learning_r = 0.1
    batch_size = 20

    training_iters=20001
    keep_prob = 0.7
    display_step = 200

    traing_num = 37884

    embedding_data_path = "../data/glove.6B.100d.txt"
    file_path = "../data/all.res"
    model_path = "../saved_model/"
    export_path = "../export_model/"


def create_model(config,session,glove,dic):#
    model = labelingModel(config,glove,dic)#
    ckpt = tf.train.get_checkpoint_state(config.model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():
    print("training model...")
    conf = LBconfig()
    brown = DataSet(conf)
    conf.vocab_size = len(brown.vocab)
    conf.tag_size = len(brown.tagSet)
    conf.rtag_size = len(brown.rtagSet)

    print("vocab_size:\t"+str(len(brown.vocab)))
    print("tag_size:\t"+str(len(brown.tagSet)))
    print("rtag_size:\t"+str(len(brown.rtagSet)))
    print("sents_size:\t"+str(len(brown.words)))

    l_r = conf.learning_r
    with tf.Session() as sess:
        model = create_model(conf,sess,brown.glove,brown.dic)#
        iters = 1   #conf.training_iters
        rg = 0
        while iters!=conf.training_iters:
            if rg+conf.batch_size > conf.traing_num:
                rg=0
            rang = range(rg,rg+conf.batch_size)
            rg+=conf.batch_size
            b_w,b_t,b_rt,b_cw,seq_len=brown.getdata(rang,conf.batch_size,conf.single_word_len)
            loss = model.step(sess, b_w, b_t,b_rt, b_cw, seq_len, conf, 2)

            if iters % conf.display_step == 0:
                print("Iter " + str(iters) + ", Minibatch Loss= " + "{:.6f}".format(loss))#+ ", Training Accuracy= " + "{:.5f}".format(acc))

                testinc=0
                testinc9 = 0
                testall=0
                testres=[]
                iinc = 0
                iinca=0
                good = {}
                bad = {}
                for ti in xrange(conf.traing_num,len(brown.words)):
                #for ti in xrange(6578,7685):
                    #testall+=1
                    conf.batch_size = 1
                    batch_xs = brown.words[ti:ti+1]
                    batch_ys = brown.tags[ti:ti+1]
                    batch_yys = brown.rtags[ti:ti+1]
                    b_w,b_t,b_rt,b_cw,seq_len = brown.normData(batch_xs,batch_ys,batch_yys,conf.single_word_len)

                    inc,inca,loss,sw,aw,gd,bd = model.step(sess, b_w, b_t, b_rt,b_cw, seq_len, conf, 1)

                    #if np.allclose(acc,1.0):#abs(acc-1.0) < 0.00001:#1.0e-9#acc==1.0:
                    #    testinc+=1
                    testinc+=inc
                    testall+=inca
                    iinc+=sw
                    iinca+=aw

                    for ii in gd:
                        if good.has_key(ii):
                            good[ii]+=gd[ii]
                        else:
                            good[ii]=gd[ii]
                    for ii in bd:
                        if bad.has_key(ii):
                            bad[ii]+=bd[ii]
                        else:
                            bad[ii]=bd[ii]

                    #if acc >=0.9:
                    #    testinc9+=1
                    #testres.append(acc)
                    #testlen.append(seq_len[0])
                    #print("test result: " + "{:.5f}".format(acc))
                print("test result(avg): " + "{:.3f}".format(testinc/float(testall))+ " " +str(testinc)+ " : " +str(testall)+" - " + "{:.3f}".format(iinc/float(iinca))+ " " +str(iinc)+ " : " +str(iinca))
                for ii in bad:
                    if not good.has_key(ii):
                        good[ii]=0
                for idd,ii in enumerate(good):
                    gd = good[ii]
                    bd = bad[ii] if bad.has_key(ii) else 0
                    print("{:>2}".format(str(idd)) + ":" + "{:>24}".format(brown.IdTotag[ii])+" "+"{:>4}".format(str(gd))+" : "+"{:<4}".format(str(bd))+" - "+"{:.3f}".format(gd/float(bd+gd)))
                

                conf.learning_r = l_r/(1.0+iters/conf.display_step*0.2)
                print(conf.learning_r)
                model.saver.save(sess,conf.model_path+"model.ckpt"+str(iters))
                #builder = saved_model_builder.SavedModelBuilder(conf.model_path)
            conf.batch_size=20
            iters+=1
        model.export(sess,conf)

    print('Done training!')


def test():
    conf = LBconfig()
    brown = DataSet(conf)
    conf.vocab_size = len(brown.vocab)
    conf.tag_size = len(brown.tagSet)
    conf.rtag_size = len(brown.rtagSet)
    input_data = brown.words
    with tf.Session() as sess:
        model = create_model(conf,sess,brown.glove)
        batch_xs = [[brown.Dic[i] if brown.Dic.has_key(i) else brown.Dic['<unk>'] for i in input_data]]
        batch_ys = [[0 for i in input_data]]
        b_w,b_t,b_cw,seq_len,b_add = brown.normData(batch_xs,batch_ys,conf.single_word_len)
        f_o = model.step(sess,b_w,b_t,b_cw,b_add,seq_len,conf.batch_size,conf,0)
    return [brown.IdToTag[i] for i in f_o]


if __name__=="__main__":
    train()
