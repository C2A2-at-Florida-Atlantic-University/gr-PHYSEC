import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Flatten, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def divisible_random(a,b,n):
    if b-a < n:
        raise Exception('{} is too big'.format(n))
        # return a
    result = random.randint(a, b)
    while result % n != 0:
        result = random.randint(a, b)
    return result

def resblock(x, kernelsize, filters, first_layer = False):
    reg = l2(0.001)  # Define L2 regularizer
    if first_layer:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
        fx = BatchNormalization()(fx)
        x = Conv2D(filters, 1, padding='same', kernel_regularizer=reg)(x)
        fx = BatchNormalization()(fx)
        out = Add()([x,fx])
        out = ReLU()(out)
    else:
        fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
        fx = BatchNormalization()(fx)
        fx = ReLU()(fx)
        fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
        fx = BatchNormalization()(fx)
        out = Add()([x,fx])
        out = ReLU()(out)

    return out 

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

class TripletNet_Channel():
    def __init__(self):
        pass
        
    def create_triplet_net(self, embedding_net, alpha):
        self.alpha = alpha
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model

    def triplet_loss_KDR(self,x):
        # Triplet Loss function.
        anchor,positive,negative = x
        anchor = self.quantization_layer(anchor)
        positive = self.quantization_layer(positive)
        negative = self.quantization_layer(negative)
        # Creating bool feature vectors for logical XOR
        zeros = tf.zeros_like(anchor)
        ones = tf.ones_like(anchor)
        #GET BOOLEAN VALUES       
        anchor_bool = tf.greater_equal(anchor,1)
        positive_bool = tf.greater_equal(positive,1)
        negative_bool = tf.greater_equal(negative,1)
        #XOR(ANCHOR,POSITIVE) , XOR(ANCHOR,NEGATIVE)
        pos_xor_bool = tf.math.logical_xor(anchor_bool,positive_bool)
        pos_xor_not_bool = tf.math.logical_not(pos_xor_bool)
        neg_xor_bool = tf.math.logical_xor(anchor_bool,negative_bool)
        neg_xor_not_bool = tf.math.logical_not(neg_xor_bool)
        #TRANSFORM BOOLEAN TO BINARY
        pos_xor = tf.where(pos_xor_bool, anchor, zeros)
        pos_xor = tf.where(pos_xor_not_bool, pos_xor, ones)
        neg_xor  = tf.where(neg_xor_bool, anchor, zeros)
        neg_xor = tf.where(neg_xor_not_bool, neg_xor, ones)
        #GET POSITIVE AND NEGATIVE MEAN (SUM(VALUES)/LENGTH(VALUES))
        pos_kdr = K.mean(pos_xor,axis=1)
        neg_kdr = K.mean(neg_xor,axis=1)
        #Calculate Loss
        basic_loss = (pos_kdr-neg_kdr+self.alpha)
        loss = K.maximum(basic_loss,0.0)
        return loss

    def triplet_loss(self,x):
        # Triplet Loss function.
        anchor,positive,negative = x
        # K.l2_normalize
        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)
        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)
        basic_loss = (pos_dist-neg_dist) + self.alpha
        loss = K.maximum(basic_loss,0.0)
        return loss  

    def quantization_layer(self,x):
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        x_mean = K.mean(x)
        x_less = K.less(x,x_mean)
        x_greater = K.greater_equal(x,x_mean)
        x_q = tf.where(x_greater, x, ones)
        x_q = tf.where(x_less, x_q, zeros)
        return x_q
    
    def feature_extractor(self, datashape):
        self.datashape = datashape
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        x =  Dropout(0.3)(x)
        x = resblock(x, 3, 32)
        x = resblock(x, 3, 32)
        #x = resblock(x, 3, 32)
        x = resblock(x, 3, 64, first_layer = True)
        x = resblock(x, 3, 64)
        #x = resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        #x = Dense(512,kernel_regularizer='l1_l2')(x)
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model             
    
    def create_generator_channel(self, batchsize, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        while True:
            list_a = []
            list_p = []
            list_n = []
            idx = divisible_random(0,len(data)-4,4)
            #batchsize_limit = batchsize+idx-1
            #print("batchsize_limit",batchsize_limit)
            #print("idx",idx)
            batch = 0
            while batch < batchsize:
                decision = bool(random.getrandbits(1))
                if decision:
                    a =data[idx]
                    n =data[idx+1]
                    p =data[idx+2]
                    list_a.append(a)
                    list_p.append(p)
                    list_n.append(n)
                else:
                    a =data[idx+2]
                    n =data[idx+3]
                    p =data[idx]
                    list_a.append(a)
                    list_p.append(p)
                    list_n.append(n)
                idx = divisible_random(0,len(data)-4,4)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            label = np.ones(int(batchsize))
            #print("label",label.shape)
            #print("A",A.shape)
            yield [A, P, N], label 

class QuadrupletNet_Channel():
    def __init__(self):
        pass
        
    def create_quadruplet_net(self, embedding_net, alpha, beta, gamma):
        
        # embedding_net = encoder()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_4 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N1 = embedding_net(input_3)
        N2 = embedding_net(input_4)
        
        loss = Lambda(self.quadruplet_loss)([A, P, N1, N2]) 
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=loss)
        return model

    # Quadruplet Loss function.
    def quadruplet_loss(self,x):
        # getting Quadruplet
        anchor,positive,negative1,negative2 = x
        # distance between the anchor and the positive features (Alice-Bob & Bob-Alice)
        D1 = K.sum(K.square(anchor-positive),axis=1)
        # distance between the anchor and negative 1 features (Alice-Bob & Alice-Eve)
        D2 = K.sum(K.square(anchor-negative1),axis=1)
        # distance between the anchor and negative 2 features (Bob-Alice & Bob-Eve)
        D3 = K.sum(K.square(positive-negative2),axis=1)
        # distance between the negative 1 and negative 2 features (Alice-Eve & Bob-Eve)
        D4 = K.sum(K.square(negative1-negative2),axis=1)
        loss = K.maximum(D1 - D2 + self.alpha,0.0) + K.maximum(D1 - D3 + self.beta,0.0) + K.maximum(D1 - D4 + self.gamma,0.0)
        return loss  

    def quantization_layer(self,x):
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        x_mean = K.mean(x)
        x_less = K.less(x,x_mean)
        x_greater = K.greater_equal(x,x_mean)
        x_q = tf.where(x_greater, x, ones)
        x_q = tf.where(x_less, x_q, zeros)
        return x_q
    
    def feature_extractor(self, datashape):
        self.datashape = datashape
        reg = l2(0.001)  # Define L2 regularizer
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
        x =  Dropout(0.3)(x)
        x = resblock(x, 3, 32, first_layer = True)
        for _ in range(3):
            x = resblock(x, 3, 32)
        x = resblock(x, 3, 64, first_layer = True)
        for _ in range(3):
            x = resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        outputs = Dense(units=512, activation='sigmoid', kernel_initializer="lecun_normal")(outputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model             
            
    def create_generator_channel(self, batchsize, data, label):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        while True:
            list_a = []
            list_p = []
            list_n1 = []
            list_n2 = []
            idx = divisible_random(0,len(data)-4,4)
            #batchsize_limit = batchsize+idx-1
            #print("batchsize_limit",batchsize_limit)
            #print("idx",idx)
            batch = 0
            while batch < batchsize:
                a =data[idx]
                n1 =data[idx+1]
                p =data[idx+2]
                n2 =data[idx+3]
                list_a.append(a)
                list_p.append(p)
                list_n1.append(n1)
                list_n2.append(n2)
                idx = divisible_random(0,len(data)-4,4)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N1 = np.array(list_n1, dtype='float32')
            N2 = np.array(list_n2, dtype='float32')
            label = np.ones(int(batchsize))
            #print("label",label.shape)
            #print("A",A.shape)
            yield [A, P, N1, N2], label 
