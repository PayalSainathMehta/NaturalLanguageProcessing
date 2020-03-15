import tensorflow as tf
import numpy as np
import math

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================
    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
    Write the code that calculate A = log(exp({u_o}^T v_c))
    A =
    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})
    B =
    ==========================================================================
    """
    
    print("Cross Entropy loss function:- ")
    
    #Code to calculate A = log(exp({u_o}^T v_c)})
    result_vector = tf.matmul(true_w,tf.transpose(inputs))
    A = tf.matrix_diag_part(result_vector)
    
    #Code to calculate B = log(\sum{exp({u_w}^T v_c)})
    exponent_vector = tf.exp(result_vector)
    sum_exponent_vector = tf.reduce_sum(exponent_vector,1)
    B = tf.log(sum_exponent_vector + 1e-5)

    # returning the loss value.
    return tf.subtract(B,A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================
    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimension is [Vocabulary].
    Implement Noise Contrastive Estimation Loss Here
    ==========================================================================
    """
    
    print("Noise Contrastive Estimation loss function:- ")
    
    
    """
    ===========================================================================================================================
                                        PROBABILITIES FOR POSITIVE SAMPLES
    ===========================================================================================================================
    """
    #Calculating the target words by looking up the weights table with id = labels
    uo = tf.nn.embedding_lookup(weights, labels)
    
    #Inputs provided are the context words.
    uc = inputs
    
    #Since we need to multiply uc and u0, we reshape u0.
    _,column = weights.get_shape()
    uo = tf.reshape(uo,[-1,column])
    #print(uo.shape)
    #print(uc.shape)
    
    #Calculating bias for the positive word vectors.
    #bo_pos is the bias we'll add to the matmul of inputs and positive values matrix.
    bo_pos = tf.nn.embedding_lookup(biases, labels)
    #We calculate u_pos = uo * T uc
    u_pos = tf.matmul(uo,tf.transpose(uc))
    #print(u_pos.shape)
    #print(bo_pos.shape)
    #adding bias to u_pos
    wo = tf.add(u_pos,bo_pos)
    #print(wo.shape)
    
    #For probabilities, we lookup the unigram tensor with id = labels.
    #Before doing that, we need to convert unigram probabilities to tensor.
    unigram_tensor = tf.convert_to_tensor(unigram_prob)
    unigram_lookup = tf.gather(unigram_tensor, labels)
    #Size of the negative samples
    k = len(sample)
    #Here we take a scalar multiplication to avoid the float to int error.
    #Calculating log(k*P(wo))
    wo_prob = tf.scalar_mul(k,unigram_lookup)
    wo_prob_log = tf.log(wo_prob)
    #print(wo.shape)
    #print(w0_prob_log.shape)
    #Taking the difference between wo and wo prob to calculate s(wo,wc) - log(k * P(wo)), where log(k * P(wo)) = wo_prob_log
    s1 = tf.subtract(wo,wo_prob_log)
    #Finally we get the positive probabilities as the sigmoid function of the above subtraction we did.
    positive_probs = tf.sigmoid(s1)
    #Log of probabilities for positive samples.
    positive_probs_log = tf.log(positive_probs + 1e-14)
    #print(positive_probs_log.shape)

    
    
    """
    ===========================================================================================================================
                                        PROBABILITIES FOR NEGATIVE SAMPLES
    ===========================================================================================================================
    """
    #Now we begin with calculating the negative probabilities
    
    #ux is calculating by the lookup of negative ids in the weights matrix
    ux = tf.nn.embedding_lookup(weights,sample)
    #print(ux.shape)
    
    #Similarly bias for negative samples is calculated by the lookup of sample ids.
    bo_neg = tf.nn.embedding_lookup(biases,sample)
    #print(bo_neg.shape)
    #print(ux.shape)
    
    #Calculating u_neg = ux * T uc
    u_neg = tf.matmul(ux,tf.transpose(uc))
    
    #Adding bias to u_neg
    wx = tf.add(tf.transpose(u_neg),bo_neg)
    
    #We use the unigram tensor we defined for positive lookups and then look it up with negative ids - sample
    unigram_negative_lookup = tf.gather(unigram_tensor,sample) 
    #Calculating probabilities by multiplying with k.
    wx_prob = tf.scalar_mul(k,unigram_negative_lookup)
    #And then taking the log of probabilities.
    wx_prob_log = tf.log(wx_prob)
    
    #Taking the difference between wx and wx prob to calculate s(wx,wc) - log(k * P(wx)), where log(k * P(wx)) = wx_prob_log
    s2 = tf.subtract(wx,wx_prob_log)
    
    #Finally we get the negative probabilities as the sigmoid function of the above subtraction we did.
    negative_probs = tf.sigmoid(s2)   
    
    #Log of probabilities for negative samples.
    negative_probs_log = tf.log(1 - negative_probs + 1e-14)
    
    summation = tf.reduce_sum(negative_probs_log,1)
    
    #print(summation.shape)
    
    """
        Now we calculate the final result by adding the positive probabilities with the summation of negative ones. And we take the negative of it.
    """
    final_samples = tf.add(- positive_probs_log, - summation)
    return final_samples
    
    
    