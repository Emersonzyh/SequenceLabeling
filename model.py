import tensorflow as tf
from tensorflow import constant
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from DataSet_pos import DataSet
import numpy as np

class labelingModel:
    def __init__(self, conf,glove,dic):

        epsilon = 1e-5
        embedding_init = tf.constant(glove)
        self.x = tf.placeholder(dtype=tf.int32,shape=[None,None],name="input_words") #batch_size * max_time
        self.y = tf.placeholder(dtype=tf.float32,shape=[None,None],name="words_tags")  #batch_size * max_time
        self.yr = tf.placeholder(dtype=tf.float32,shape=[None,None],name="words_rough_tags") #batch_size * max_time
        self.x_c = tf.placeholder(dtype=tf.int32,shape=[None,None,conf.single_word_len],name="input_words_character") #batch_size * max_time * single_word_len
        self.sequence_length = tf.placeholder(dtype=tf.int32,shape=[None],name="sequence_length") #batch_size
        self.keep_prob = tf.placeholder("float")
        self.l_rate = tf.placeholder("double")
        self.max_size = tf.shape(self.x)[1]
        self.batch_tensor = tf.shape(self.x)[0]
        self.dic = dic
        #CNN part
        with tf.variable_scope("embedding_part"):
            embedding_words = tf.Variable(embedding_init,name="embedding_weights_words",dtype=tf.float32)
            embedding_chars = tf.get_variable(name="embedding-weights_character",initializer=tf.truncated_normal([conf.char_num,conf.n_input]),dtype=tf.float32)
            input_word_vector = tf.nn.embedding_lookup(embedding_words,self.x,name="word_vector")  #batch_size * max_time * n_input
            input_word_matrix = tf.nn.embedding_lookup(embedding_chars,self.x_c,name="charcter_vector") #batch_size * max_time * single_word_len * n_input

        with tf.variable_scope("character_CNN"):
            input_word_matrix_r = tf.reshape(input_word_matrix,[self.batch_tensor*self.max_size,conf.single_word_len,conf.n_input])
            input_word_matrix_c = tf.expand_dims(input_word_matrix_r,axis=-1)
            
            filte = tf.Variable(tf.random_normal([5,5,1,1]),name='filter')
            input_word_matrix_dp = tf.nn.dropout(x=input_word_matrix_c,keep_prob=self.keep_prob,name="CNN_input_Dropout")

            input_conv=tf.nn.conv2d(input_word_matrix_dp,filte,strides=[1,1,1,1],padding = "SAME",data_format="NHWC")
            input_pool=tf.nn.max_pool(input_conv,ksize=[1,5,5,1],strides=[1,5,5,1],padding = 'SAME',data_format="NHWC")    
            output_conv = tf.squeeze(input_pool,axis=-1)  #batch_size * max_time * single_word_len/5 * n_input/5

            print "output_conv",output_conv.get_shape()

        #LSTM part1
        with tf.variable_scope("LSTM_Part1"):
            lstm_input = tf.reshape(output_conv,[self.batch_tensor,self.max_size,conf.single_word_len*conf.n_input/25])
            print lstm_input
            lstm_cell_b={}
            def lstm_cell(hidden_size):
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
                drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return drop
            for direction in ["forward","backward"]:
                with tf.variable_scope(direction):
                    lstm_cell_b[direction] = tf.contrib.rnn.MultiRNNCell([lstm_cell(conf.n_hidden) for _ in range(conf.num_layers)], state_is_tuple=True)
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_b["forward"],
                                                        lstm_cell_b["backward"],
                                                        lstm_input,
                                                        dtype=tf.float32,
                                                        sequence_length = self.sequence_length)#,
            output_forward,output_backward = outputs
            outputs_concated = tf.concat([output_forward,output_backward],axis=-1)  #batch_size * max_time * n_hidden
            print outputs_concated

        #cost 1
        with tf.variable_scope("cost_1"):
            W = tf.Variable(tf.truncated_normal([conf.n_hidden*2,conf.rtag_size]),name="First_Cost_Weight",dtype=tf.float32)
            b = tf.Variable(tf.zeros([conf.rtag_size]),name="First_Cost_bias",dtype=tf.float32)
            outputs_concated_reshape = tf.reshape(outputs_concated,[-1,conf.n_hidden*2])
            rough_output = tf.matmul(outputs_concated_reshape,W)+b
            self.rough_out = tf.reshape(tf.nn.tanh(rough_output),[self.batch_tensor,self.max_size,conf.rtag_size])

            with tf.variable_scope("softmax"):
                yr = tf.reshape(tf.one_hot(tf.cast(self.yr,tf.int32),conf.rtag_size),[-1,conf.rtag_size])
                lr = tf.nn.softmax_cross_entropy_with_logits(logits=self.rough_out, labels=yr)
                mask = tf.sequence_mask(self.sequence_length)
                losses_r = tf.boolean_mask(lr, mask)
                self.loss_r = tf.reduce_mean(losses_r,name="rough_loss")
            
        #Layer Normalization
        with tf.variable_scope("LN"):
            mean, variance = tf.nn.moments(input_word_vector, [2], keep_dims=True)
            normalised_input_vector = (input_word_vector - mean) / tf.sqrt(variance + epsilon)
            print "normalised_input_vector ",normalised_input_vector.get_shape()
            
            mean2, variance2 = tf.nn.moments(outputs_concated, [2], keep_dims=True)
            normalised_output_vector = (outputs_concated - mean2) / tf.sqrt(variance2 + epsilon)
            print "normalised_output_vector ",normalised_output_vector.get_shape()
            
            mean3, variance3 = tf.nn.moments(lstm_input, [2], keep_dims=True)
            normalised_cnn_vector = (lstm_input - mean3) / tf.sqrt(variance3 + epsilon)
            print "normalised_cnn_vector ",normalised_cnn_vector.get_shape()
            
        #LSTM part2
        with tf.variable_scope("Lstm_part2"):
            lstm_input2 = tf.concat([input_word_vector,lstm_input,self.rough_out],axis=-1)
            #lstm_input2 = input_word_vector
            lstm_input3 = tf.nn.dropout(lstm_input2,keep_prob=self.keep_prob,name="Dropout_LSTM_2")
            lstm_cell_b2={}
            for direction in ["forward","backward"]:
                with tf.variable_scope(direction):
                    lstm_cell_b2[direction] = tf.contrib.rnn.MultiRNNCell([lstm_cell(conf.n_hidden*2) for _ in range(conf.num_layers)], state_is_tuple=True)
            outputs2,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_b2["forward"],
                                                         lstm_cell_b2["backward"],
                                                         lstm_input3,
                                                         dtype=tf.float32,
                                                         sequence_length = self.sequence_length)#,
            output_forward2,output_backward2 = outputs2
            outputs_concated2 = tf.concat([output_forward2,output_backward2],axis=-1)  #batch_size * max_time * n_hidden
            print outputs_concated2.get_shape()
        #CNN_with_ALU
        '''with tf.variable_scope("CNN_with_ALU1"):
            embedding_pos = tf.Variable(embedding_init,name="embedding_weights_pos",dtype=tf.float32)
            a = []
            for i in range(500):
                if self.dic.has_key(str(i)):
                    #print(dic[str(i)])
                    if int(self.dic[str(i)])<9543:
                        a.append(int(self.dic[str(i)]))
                    else:
                        a.append(int(self.dic[str(0)]))
                else:
                    a.append(int(self.dic[str(0)]))
            pos = tf.Variable(a)
            input_pos_vector = []
            input_pos = tf.nn.embedding_lookup(embedding_pos,pos,name="pos_vector")
            input_pos_c = tf.slice(input_pos,[0,0],[self.max_size,100])
            for j in range(20):
                input_pos_vector.append(input_pos_c)
            input_pos_vector_s = tf.slice(input_pos_vector,[0,0,0],[self.batch_tensor,self.max_size,100])
            input_word_vector_c1 = tf.concat([input_word_vector,lstm_input,self.rough_out],axis=-1)
            input_word_vector_v1 = tf.concat([input_word_vector_c1,input_pos_vector_s],2)

            seq_len = tf.shape(input_word_vector_v1)[1]
            pad_s = tf.Variable(tf.random_normal([20,2,270], name=None),name="padding")
            pad = tf.slice(pad_s,[0,0,0],[self.batch_tensor,2,270])
            input_word_vector_p1 = tf.concat([pad,input_word_vector_v1],1)
            input_word_vector_p1 = tf.concat([input_word_vector_p1,pad],1)
    
            input_word_vector_r1 = tf.reshape(input_word_vector_p1,
[self.batch_tensor,seq_len+4,270])
            input_word_vector_re1 = tf.expand_dims(input_word_vector_r1,axis=-1)

            filte1 = tf.Variable(tf.random_normal([5,270,1,200]),name="filter1")
            input_conv1 = tf.nn.conv2d(input_word_vector_re1,filte1,strides=[1,1,1,1],padding = "VALID",data_format = "NHWC")
            input_conv_r1 = tf.reshape(input_conv1,
[self.batch_tensor,seq_len,200])

            #ALU_input1 = tf.slice(input_conv_r1,[0,0,0],[self.batch_tensor,seq_len,100])
            #ALU_gate1 = tf.slice(input_conv_r1,[0,0,100],[self.batch_tensor,seq_len,100])
            #ALU_output1 = tf.multiply(ALU_input1,ALU_gate1)
            #print ALU_output1.get_shape()


        with tf.variable_scope("CNN_with_ALU2"):
            
            #input_word_vector_c2 = tf.concat([ALU_output1,lstm_input,self.rough_out],axis=-1)
            #input_word_vector_v2 = tf.concat([input_word_vector_c2,input_pos_vector_s],2)
            pad_s = tf.Variable(tf.random_normal([20,2,200], name=None),name="padding")
            pad = tf.slice(pad_s,[0,0,0],[self.batch_tensor,2,200])
            input_word_vector_p2 = tf.concat([pad,input_conv_r1],1)
            input_word_vector_p2 = tf.concat([input_word_vector_p2,pad],1)
            #print(input_word_vector_p2)
            
            input_word_vector_r2 = tf.reshape(input_word_vector_p2,
[self.batch_tensor,seq_len+4,200])
            input_word_vector_c2 = tf.expand_dims(input_word_vector_r2,axis=-1)

            filte2 = tf.Variable(tf.random_normal([5,200,1,100]),name="filter2")
            input_conv2 = tf.nn.conv2d(input_word_vector_c2,filte2,strides=[1,1,1,1],padding = "VALID",data_format = "NHWC")
            input_conv_r2 = tf.reshape(input_conv2,
[self.batch_tensor,seq_len,100])

            #ALU_input2 = tf.slice(input_conv_r2,[0,0,0],[self.batch_tensor,seq_len,100])
            #ALU_gate2 = tf.slice(input_conv_r2,[0,0,100],[self.batch_tensor,seq_len,100])
            #ALU_output2 = tf.multiply(ALU_input2,ALU_gate2)
            #print ALU_output2.get_shape()'''


        '''with tf.variable_scope("CNN_with_ALU3"):
            
            #input_word_vector_c3 = tf.concat([ALU_output2,lstm_input,self.rough_out],axis=-1)
            #input_word_vector_v3 = tf.concat([input_word_vector_c3,input_pos_vector_s],2)
            pad_s = tf.Variable(tf.random_normal([20,2,100], name=None),name="padding")
            pad = tf.slice(pad_s,[0,0,0],[self.batch_tensor,2,100])
            input_word_vector_p3 = tf.concat([pad,input_conv_r2],1)
            input_word_vector_p3 = tf.concat([input_word_vector_p3,pad],1)
    
            input_word_vector_r3 = tf.reshape(input_word_vector_p3,
[self.batch_tensor,seq_len+4,100])
            input_word_vector_c3 = tf.expand_dims(input_word_vector_r3,axis=-1)

            filte3 = tf.Variable(tf.random_normal([5,100,1,100]),name="filter3")
            input_conv3 = tf.nn.conv2d(input_word_vector_c3,filte3,strides=[1,1,1,1],padding = "VALID",data_format = "NHWC")
            input_conv_r3 = tf.reshape(input_conv3,
[self.batch_tensor,seq_len,100])

            #ALU_input3 = tf.slice(input_conv_r3,[0,0,0],[self.batch_tensor,seq_len,50])
            #ALU_gate3 = tf.slice(input_conv_r3,[0,0,50],[self.batch_tensor,seq_len,50])
            #ALU_output3 = tf.multiply(ALU_input3,ALU_gate3)
            #print ALU_output3.get_shape()'''
        #cost 2
        with tf.variable_scope("cost_2"):
            W2 = tf.Variable(tf.truncated_normal([conf.n_hidden*4,conf.tag_size]),name="Second_Cost_Weight",dtype=tf.float32)
            #W2 = tf.get_variable(shape=[conf.n_hidden*8,conf.tag_size],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.0, shape=[conf.tag_size]),name="final_Cost_bias",dtype=tf.float32)
            outputs_concated2_rp = tf.reshape(outputs_concated2,[-1,conf.n_hidden*4])

            #self.final_output = tf.reshape(tf.matmul(outputs_concated2_rp,W2)+b2,[self.batch_tensor,self.max_size,conf.tag_size])
            #self.final_output = tf.matmul(outputs_concated2_rp,W2)+b2
            #yy = tf.reshape(tf.one_hot(tf.cast(self.y,tf.int32),conf.tag_size),[-1,conf.tag_size])
            self.predictions = tf.nn.xw_plus_b(outputs_concated2_rp, W2, b2, name="predictions") # input : [batch_size * timesteps , 2*n_hidden_LSTM] * [2*n_hidden_LSTM, num_classes]  = [batch_size * timesteps , num_classes]
            self.final_output = tf.reshape(self.predictions, [self.batch_tensor, self.max_size, conf.tag_size],name="logits") # output [batch_size, max_seq_len] 
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.final_output, labels=tf.cast(self.y,tf.int32))
            mask = tf.sequence_mask(self.sequence_length)
            losses2 = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses2)
            """
            with tf.variable_scope("softmax"):
                ll = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_output, labels=yy)
                #mask = tf.sequence_mask(self.sequence_length)
                #losses = tf.boolean_mask(ll, mask)
                #self.loss = tf.reduce_mean(losses,name="final_loss")
                self.loss = tf.reduce_mean(ll,name="final_loss")
            """
        #optimizer
        with tf.variable_scope("opti"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate).minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, b_w, b_t,b_rt,b_cw,seq_len, config, test):
        if test==1: # test then return accuracy and cost
            loss,final_op=session.run([self.loss,self.final_output],
                feed_dict={self.x:b_w,self.y:b_t,self.yr:b_rt,self.x_c:b_cw,self.sequence_length:seq_len,
                self.keep_prob:1.0})
            inc = 0
            inca = 0
            sw = 0
            aw = 0
            good = {}
            bad={}
            #print final_op
            final_op = np.argmax(final_op,2)
            #print final_op
            for i,j in enumerate(final_op):
                f_p =j[:seq_len[i]]
                #print f_p
                #print b_t[i][:seq_len[i]]
                ar,iinc,iinca,gd,bd = self.score_(f_p,b_t[i][:seq_len[i]])
                if ar:
                    inc+=1
                inca+=1
                sw+=iinc
                aw+=iinca
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
            return inc,inca,loss,sw,aw,good,bad
        elif test==2: # train then return accuracy and cost
            loss,_=session.run([self.loss,self.optimizer], 
                feed_dict={self.x:b_w,self.y:b_t,self.yr:b_rt,self.x_c:b_cw,self.sequence_length:seq_len,
                self.keep_prob:config.keep_prob,self.l_rate:config.learning_r})
            return loss

    def score_(self,logits,lables):
        good = {}
        bad = {}
        inc = 0
        inca = 0
        ar = True
        for i,j in zip(logits,lables):
            inca+=1
            if i!=int(j):
                if bad.has_key(j):
                    bad[j]+=1
                else:
                    bad[j]=1
                ar = False
                continue
            if good.has_key(j):
                good[j]+=1
            else:
                good[j]=1
            inc+=1
        return ar,inc,inca,good,bad
    def export(self,sess,conf):
        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["cost/transition_params","cost/Reshape_1"])  
        tf.train.write_graph(graph, conf.export_path, 'graph.pb', as_text=False) 

