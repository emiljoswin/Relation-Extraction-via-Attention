import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class MyBasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        # model = models.Sequential()
        self.forward_layer = layers.GRU(hidden_size, return_sequences=True)
        self.backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True)
        # model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer)) 

        # self.model = model

        ### TODO(Students) START
        # ...
        ### TODO(Students) END

    def attn(self, rnn_outputs):
        ### TODO(Students) START
        # ...
        ### TODO(Students) END
        
        ## rnn_outputs = bsz x seq_len x 256  - eqn 8 from the paper is not done
        H = tf.transpose(rnn_outputs, [0, 2, 1]) # bsz x 256 x seq_len
        M = tf.nn.tanh(H) # [10 x 5 x 256] , self.omegas = [256 x 1]

        # self.omegas => 256 x 1, M => bsz x 256 x seq_len

        alpha = tf.tensordot(M, tf.transpose(self.omegas), axes=[1, 1] ) # bsz x seq_len x 1
        alpha = tf.squeeze(alpha) # bsz x seq_len
        alpha = tf.nn.softmax(alpha, axis=1) # do softmax across seq_len => bsz x seq_len

        alpha = tf.expand_dims(alpha, axis=1) # bsz x 1 x seq_len
        r = tf.matmul(H, alpha, transpose_b=True) # bsz x 256 x 1
        r = tf.squeeze(r) # bsz x 256
        r = tf.nn.tanh(r)
        return r

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        ### TODO(Students) END

        tokens_mask = tf.cast(inputs!=0, tf.float32)

        x = tf.concat([word_embed, pos_embed], axis=2)
        x = word_embed

        # logits = self.model(x, mask=tokens_mask)
        x1 = self.forward_layer(x, mask=tokens_mask)
        x2 = self.backward_layer(x, mask=tokens_mask)

        x = tf.concat([x1, x2], axis=2)
        logits = self.attn(x)

        logits = self.decoder(logits)

        return {'logits': logits}

class MyAdvancedModel(models.Model):
    "This is the real advanced model that the predictions are based on"

    def __init__(self, vocab_size: int, embed_dim : int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()

        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        self.drop = layers.Dropout(0.3)
        embed_dim = 5
        seq_len = 5
        n_filters = 100 
        
        filter_size = 2 #number of words seen during one conv operation
        self.conv1 = layers.Conv1D(n_filters, filter_size)
        
        filter_size = 3 #number of words seen during one conv operation
        self.conv2 = layers.Conv1D(n_filters, filter_size)
        
        filter_size = 4 #number of words seen during one conv operation
        self.conv3 = layers.Conv1D(n_filters, filter_size)
        
        self.dropout = layers.Dropout(0.5)
        self.linear = layers.Dense(units=self.num_classes)


    def attn(self, outputs):
        pass

    def call(self, inputs, pos_inputs, training):
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        tokens_mask = tf.cast(inputs!=0, tf.float32)

        x = tf.concat([word_embed, pos_embed], axis=2)

        x = self.drop(x)
        # print(inputs.shape)
        out1 = tf.nn.tanh( self.conv1(x) ) # bsz x (filter_size - seq_len + 1) x n_filters
        out2 = tf.nn.tanh( self.conv2(x) )
        out3 = tf.nn.tanh( self.conv3(x) )
        
        # bsz x 1 x n_filters - before squeeze
        # bsz x n_filters - after squeeze
        pool1 = tf.squeeze( layers.MaxPool1D(out1.shape[1])(out1) ) 
        pool2 = tf.squeeze( layers.MaxPool1D(out2.shape[1])(out2) )
        pool3 = tf.squeeze( layers.MaxPool1D(out3.shape[1])(out3) ) 
        
        # bsz x n_filters * (number of different conv2d layers)
        x = tf.concat([pool1, pool2, pool3], axis=1)
        
        x = self.dropout(x)
        
        x = self.linear(x)

        return {'logits': x}