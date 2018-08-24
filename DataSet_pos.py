import numpy as np
import collections
import random


class DataSet:

    AA = ["JJ","JJR","JJS","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB","WP$"]
    AL_TAGS = ['PRP$', '``', 'POS', "''", 'DT', '#','RP', '$', 'FW', ',', '.', 'TO', 'PRP','-LRB-', ':', 'CC', 'LS', 'PDT', 'CD',
          'EX', 'IN', 'MD', '-RRB-', 'SYM', 'UH',
          "JJ","JJR","JJS","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WRB","WP$"]
    RH_TAGS = ['PRP$', '``', 'POS', "''", 'DT', '#','RP', '$', 'FW', ',', '.', 'TO', 'PRP','-LRB-', ':', 'CC', 'LS', 'PDT', 'CD',
          'EX', 'IN', 'MD', '-RRB-', 'SYM', 'UH',
          "J","N","R","V","W"]

    def __init__(self,conf):
        self.dic,self.glove = self.loadGlove(conf)
        
        self.words,self.tags,self.rtags,self.vocab,self.tagSet,self.rtagSet,self.CharToId = self.loadData(conf.file_path)
        self.tagToId,self.IdTotag,self.Dic,self.IdToWord,self.rtagToId,self.glove = self.vocabMap()
        self.wordsId,self.tagsId,self.rtagsId = self.DataToId()
    
    def vocabMap(self):
        tag_to_id = {j:i for i,j in enumerate(self.tagSet)}
        id_to_tag = {i:j for i,j in enumerate(self.tagSet)}
        Dic = {j:i for i,j in enumerate(self.vocab)}

        glove = []
        for i in Dic:
            if i in self.dic:
                glove.append(self.glove[self.dic[i]])
            else:
                glove.append([0.0 for j in range(len(self.glove[0]))])
        for i in range(500):
            glove.append(self.glove[self.dic[str(i)]])
        ToDic = {i:j for i,j in enumerate(self.vocab)}
        rtag_to_id = {j:i for i,j in enumerate(self.rtagSet)}
        return tag_to_id,id_to_tag,Dic,ToDic,rtag_to_id,glove

    def findId(self,ind,dic):
        if dic.has_key(ind):
            return dic[ind]
        else:
            return dic['<unk>']

    def DataToId(self):
        wordsId=[[self.findId(i,self.Dic) for i in j] for j in self.words]
        tagId = [[self.findId(i,self.tagToId) for i in j] for j in self.tags]
        rtagId = [[self.findId(i,self.rtagToId) for i in j] for j in self.rtags]
        return wordsId,tagId,rtagId

    def IdToData(self,ids):
        labels = [self.IdTotag[i] for i in ids]
        return labels

    def loadData(self,fpath):
        fi = open(fpath,'r')
        words = []
        tags = []
        rtags = []
        vocab = {}
        words_tmp = []
        tags_tmp = []
        rtags_tmp = []
        for i in fi.readlines():
            a = i.strip("\n").split()
            if len(a)!=2:
                continue
            words_tmp.append(a[0].lower())
            tags_tmp.append(a[1])
            if a[1] in DataSet.AA:
                rtags_tmp.append(a[1][0]) 
            else:
                rtags_tmp.append(a[1])
            if a[1]==".":
                words.append(words_tmp)
                tags.append(tags_tmp)
                rtags.append(rtags_tmp)
                words_tmp = []
                tags_tmp = []
                rtags_tmp = []
            if len(words) < 37884:
                if a[0].lower() in vocab:
                    vocab[a[0].lower()]+=1
                else:
                    vocab[a[0].lower()]=1
        vocab_tmp = [i[0] for i in sorted(vocab.items(), lambda x, y: cmp(x[1], y[1]),reverse=True) if i[1]>5]
        vocab_tmp.append("<unk>")

        chars=set(['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87', '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f', '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97', '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f', '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf', '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7', '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7', '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xce', '\xcf', '\xd0', '\xd1', '\xd7', '\xd8', '\xd9', '\xda', '\xdb', '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7', '\xe8', '\xe9', '\xeb', '\xec', '\xed', '\xef'])
        chs={j:i+1 for i,j in enumerate(chars)}
        chs[' ']=0
        return words,tags,rtags,vocab_tmp,DataSet.AL_TAGS,DataSet.RH_TAGS,chs

    def getdata(self,use_size,batch_size,single_word_len):
        rand = use_size
        c_ws=[]
        ws = [self.wordsId[i] for i in rand]
        ts = [self.tagsId[i] for i in rand]
        rts = [self.rtagsId[i] for i in rand]

        seq_len = [len(self.wordsId[i]) for i in rand]
        
        max_len = max(seq_len)
        for i in range(batch_size):
            while len(ws[i])<max_len:
                ws[i].append(self.Dic["."])
                ts[i].append(self.tagToId["."])
                rts[i].append(self.rtagToId["."])

        for i in range(len(ws)):
            ar=[]
            for j in ws[i]:
                k=[0 for kk in range(single_word_len)]
                m=len(self.IdToWord[j])
                if m>single_word_len:
                    for kk in range(single_word_len):
                        k[kk]=self.CharToId[self.IdToWord[j][m-single_word_len+kk]]
                else:
                    for kk in range(m):
                        k[(single_word_len-m)/2+kk]=self.CharToId[self.IdToWord[j][kk]]
                ar.append(k)
            c_ws.append(ar)
        return np.array(ws,dtype=np.int32),np.array(ts,dtype=np.float32),np.array(rts,dtype=np.float32),np.array(c_ws,dtype=np.int32),np.array(seq_len,dtype=np.int32)

    def normData(self,words,tags,rtags,single_word_len):
        words = [[self.findId(i,self.Dic) for i in j] for j in words]
        tags = [[self.findId(i,self.tagToId) for i in j] for j in tags]
        rtags = [[self.findId(i,self.rtagToId) for i in j] for j in rtags]
        seq_len=[len(i) for i in words]

        max_len = max(seq_len)
        for i in range(len(words)):
            while len(words[i])<max_len:
                words[i].append(self.Dic["."])
                tags[i].append(self.tagToId["."])
                rtags[i].append(self.rtagToId["."])

        c_ws=[]
        for i in range(len(words)):
            ar=[]
            for j in words[i]:
                k=[0 for kk in range(single_word_len)]
                m=len(self.IdToWord[j])
                if m>single_word_len:
                    for kk in range(single_word_len):
                        k[kk]=self.CharToId[self.IdToWord[j][m-single_word_len+kk]]
                else:
                    for kk in range(m):
                        k[(single_word_len-m)/2+kk]=self.CharToId[self.IdToWord[j][kk]]
                ar.append(k)
            c_ws.append(ar)
        
        return np.array(words,dtype=np.int32),np.array(tags,dtype=np.float32),np.array(rtags,dtype=np.float32),np.array(c_ws,dtype=np.int32),np.array(seq_len,dtype=np.int32)

    def loadGlove(self,config):
        e_d_p = open(config.embedding_data_path,"r")
        dic = {}
        glove = []
        j = 0
        for i in e_d_p.readlines():
            a=i.split()
            dic[a[0].lower()]=j
            j+=1
            glove.append([float(k) for k in a[1:]])
            if j%100000==0:
                print j
                break
        return dic,glove

