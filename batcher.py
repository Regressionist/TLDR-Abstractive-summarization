import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import data
import random
random.seed(1234)

class Example(object):
    def __init__(self, article, abstract_sentences, vocab):
        start_decoding=vocab.word2id(data.START_DECODING)
        stop_decoding=vocab.word2id(data.STOP_DECODING)
        
        article_words=article.split()
        if len(article_words)>config.max_enc_steps:
            article_words=article_words[:config.max_enc_steps]
        self.enc_len=len(article_words)
        self.enc_input=[vocab.word_to_id(w) for w in article_words]
        
        abstract = ' '.join(abstract_sentences)
        abstract_words = abstract.split()
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len=len(self.dec_input)
        
        if config.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)
            
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences
        
    
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp=[start_id]+sequence[:]
        target=sequence[:]
        if len(inp)>max_len:
            inp=inp[:max_len]
            target=target[:max_len]
        else:
            target.append(stop_id)
        return inp, target
    
    def pad_decoder_input_target(self,pad_id,max_len):
        while(len(self.dec_input)<max_len):
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)
            
    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if config.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)
                
class Batch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size=batch_size
        self.pad_id=vocab.word_to_id(data.PAD_TOKEN)
        self.init_encoder_seq()
        self.init_decoder_seq()
        self.store_orig_srings(example_list)
        
    def init_encoder_seq(self, example_list):
        max_enc_seq_len=max([ex.enc_len for ex in example for ex in example list])
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
        
        self.enc_batch=np.zeros((self.batch_size,max_enc_seq_len), dtype=np.int32)
        self.enc_lens=np.zeros((self.batch_size),dtype=np.int32)
        self.enc_padding_mask=np.zeros((self.batch_size,max_enc_seq_len), dtype=np.float32)
        
        for i,ex in enumerate(example_list):
            self.enc_batch[i,:]=ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in xrange(ex.enc_len):
                self.enc_padding_mask[i][j] = 1
        if config.pointer_gen:
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            self.art_oovs = [ex.article_oovs for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]
                
    def init_decoder_seq(self, example_list):
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)
        
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)
        
        for i,ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in xrange(ex.dec_len):
                self.dec_padding_mask[i][j] = 1
                
    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list] # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists
        

        
class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""
    def __init__(self, data_path, vocab, mode, batch_size, single_pass, batch_queue_max):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self.BATCH_QUEUE_MAX=
        self._batch_queue=Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue=Queue.Queue(self.BATCH_QUEUE_MAX*self.batch_size)
        if single_pass:
            self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16 # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing
        
        self._example_q_threads = []
        for _ in xrange(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
            self._batch_q_threads = []
        for _ in xrange(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()
        if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()
            
    def next_batch(self):
        batch = self._batch_queue.get() # get the next Batch
        return batch
    
    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
        while True:
            try:
                (article, abstract) = input_gen.next() # read the next example from file. article and abstract are both strings.
            except StopIteration: # if there are no more examples:
            if self._single_pass:
                self._finished_reading = True
                break
            else:
                raise Exception("single_pass mode is off but the example generator is out of data; error.")
                
            abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = Example(article, abstract_sentences, self._vocab) # Process into an Example.
            self._example_queue.put(example) # place the Example in the example queue.
            
    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
        # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in xrange(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
            else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in xrange(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in xrange(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
                    
    def text_generator(self, example_generator):
        while True:
            e = example_generator.next()
            try:
                article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
            except ValueError:
                continue
            if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        #tf.logging.warning('Found an example with empty article text. Skipping it.')
                continue
            else:
                yield (article_text, abstract_text)