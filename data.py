import glob
import random
import struct


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences


class Vocab(object):
    def __init__(self, vocab_file, max_size):
        self.word2id={}
        self.id2word={}
        self.count=0
        
        for word in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self.word2id[word]=self.count
            self.id2word[self.count]=word
            self.count=self.count+1
            
        with open(vocab_file,'r') as f:
            for line in f:
                pieces=line.split()
                if len(pieces)!=2:
                    print ('Incorrectly formatted line in vocab file')
                w=pieces[0]
                if w not in self.word2id:
                    self.word2id[w]=self.count
                    self.id2word[self.count]=w
                    self.count=self.count+1
                
                if self.count>=max_size:
                    print ('Max size of Vocab reached; stopped reading!')
                   
        print('Finished with the vocbulary; Last word added:{}'.format(self.id2word[max_size-1]))
        
        
    def word_to_id(self,word):
        if word not in self.word2id:
            return self.word2id[UNKNOWN_TOKEN]
        return self.word2id[word]
    
    def id_to_word(self,idx):
        if idx not in self.id2word:
            return UNKNOWN_TOKEN
        return self.id2word[idx]
    
    def size(self):
        return self.count
    
    def write_metadata(self,path):
        with open(path,'w') as f:
            fieldnames=['word']
            writer=csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self.id2word[i]})
                
def example_generator(path,single_pass):
    while True:
        filelist = glob.glob(path)
        if single_pass:
            file_list=sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader=open(f,'rb')
            len_bytes = reader.read(8)
            if not len_bytes:
                break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print ("example_generator completed reading all datafiles. No more data.")
            break

def article2ids(article_words,vocab):
    ids=[]
    oovs=[]
    unk_id = vocab.word_to_id(UNKNOWN_TOKEN)
    for w in article_words:
        i=vocab.word_to_id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    ids=[]
    unk_id=vocab.word_to_id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i=vocab.word_to_id(w)
        if i==unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
    words=[]
    for i in id_list:
        try:
            w = vocab.id_to_word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words

def abstract2sents(abstract):
    cur=0
    sents=[]
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e:
            return sents
        
def show_art_oovs(article, vocab):
    unk_token = vocab.word_to_id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str

def show_abs_oovs(abstract, article_oovs, vocab):
    unk_token = vocab.word_to_id(UNKNOWN_TOKEN)
    words=abstract.split(" ")
    new_words=[]
    for w in words:
        if vocab.word2id(w) == unk_token:
            if article_oovs is None:
                new_words.append("__%s__" % w)
            else:
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str