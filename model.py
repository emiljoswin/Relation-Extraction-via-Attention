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

        model = models.Sequential()
        forward_layer = layers.GRU(hidden_size, return_sequences=True)
        backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True)
        model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(5, 2*embed_dim))) 
        
        self.model = model

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

        logits = self.model(x, mask=tokens_mask)

        logits = self.attn(logits)

        logits = self.decoder(logits)

        return {'logits': logits}


class MyAdvancedModel(models.Model):

    def __init__(self, vocab_size: int, embed_dim : int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        # ...
        ### TODO(Students END
        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.drop = layers.Dropout(0.3)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        layer1 = models.Sequential()
        forward_layer = layers.GRU(hidden_size, return_sequences=True, dropout=0.3) # TODO -emil - try other initializers
        backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True, dropout=0.3)
        layer1.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(5, 2*embed_dim))) 

        layer2 = models.Sequential()
        forward_layer = layers.GRU(hidden_size, return_sequences=True, dropout=0.3) # TODO -emil - try other initializers
        backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True, dropout=0.3)
        layer2.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(5, 2*hidden_size))) 

        layer3 = models.Sequential()
        forward_layer = layers.GRU(hidden_size, return_sequences=True, dropout=0.5) # TODO -emil - try other initializers
        backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True, dropout=0.5)
        layer3.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(5, 2*hidden_size))) 

        layer4 = models.Sequential()
        forward_layer = layers.GRU(hidden_size, return_sequences=True, dropout=0.5) # TODO -emil - try other initializers
        backward_layer = layers.GRU(hidden_size, return_sequences=True, go_backwards=True, dropout=0.5)
        layer4.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(5, 2*hidden_size))) 

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def attn(self, rnn_outputs):
        
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
        ### TODO(Students) START
        # ...
        ### TODO(Students END
        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        ### TODO(Students) START
        # ...
        ### TODO(Students) END
        # print('training', training)
        tokens_mask = tf.cast(inputs!=0, tf.float32)

        x = tf.concat([word_embed, pos_embed], axis=2)

        logits = self.layer1(x, mask=tokens_mask, training=training)

        logits1 = self.layer2(logits, mask=tokens_mask, training=training)

        residual = logits + logits1

        logits = self.layer3(residual, mask=tokens_mask, training=training)

        logits1 = self.layer4(logits, mask=tokens_mask, training=training)

        residual = logits + logits1

        logits = self.attn(residual)

        if training:
            logits = self.drop(self.decoder(logits))
        else:
            logits = self.decoder(logits)

        return {'logits': logits}