import os, sys
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
import operator
from operator import itemgetter

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
if sys.argv[1] == 'word_analogy_dev':
    input_file = open('word_analogy_dev.txt', 'r')
    if loss_model == 'cross_entropy':
        output_file = open('word_analogy_dev_predictions_cross_entropy.txt','w')
    else:
        output_file = open('word_analogy_dev_predictions_nce.txt', 'w')
elif sys.argv[1] == 'word_analogy_test':
    input_file = open('word_analogy_test.txt', 'r')
    if loss_model == 'cross_entropy':
        output_file = open('word_analogy_test_predictions_cross_entropy.txt','w')
    else:
        output_file = open('word_analogy_test_predictions_nce.txt', 'w')

result = ""
#result2 = ""

for line in input_file:
    line.strip()
    words_left_diff_vectors = []
    words_right_diff_vectors = []
    word_pairs_left,word_pairs_right = line.split("||")
    words_left = word_pairs_left.replace('"','').replace('\n','').split(",")
    #print(words_left)
    words_right = word_pairs_right.replace('"','').replace('\n','').split(",")
    #print(word_right)
    for word in words_left:
        word1 = word.split(":")[0]
        word2 = word.split(":")[1]
        words_left_diff_vectors.append(embeddings[dictionary[word2]]-embeddings[dictionary[word1]])
       
    for word in words_right:
        word1 = word.split(":")[0]
        word2 = word.split(":")[1]
        words_right_diff_vectors.append(embeddings[dictionary[word2]]-embeddings[dictionary[word1]])
     
    words_left_diff_vectors = np.array(words_left_diff_vectors)
    words_right_diff_vectors = np.array(words_right_diff_vectors)
    
    words_left_diff_vectors = np.mean(words_left_diff_vectors,axis = 0)
    #print(words_left_diff_vectors.shape)
    similarity = []
    for word in words_right_diff_vectors:
        cosine_similarity = dot(word,words_left_diff_vectors)/(norm(word)*norm(words_left_diff_vectors))
        similarity.append(cosine_similarity)
    
    
    #print(similarity)
    max = np.where(similarity == np.amax(similarity))
    min = np.where(similarity == np.amin(similarity))

    #for each word in choices, we now append the least similar and the most similar ones.
    result = "\" ".join(["\""+word for word in words_right])+"\""
    result = result+' \"'+words_right[min[0][0]]+'\"'
    result = result+' \"'+words_right[max[0][0]]+'\"'
    
   
    output_file.write(result+ '\n')

"""
============================================================================================================================
                                            FINDING TOP 20 SIMILAR WORDS 
============================================================================================================================
"""
#For finding the top 20 similar words to below words
word_list = ['american','first','would']

#First we iterate in our list of 3 words and store their embeddings by looking up the "embeddings" list we created as per our model.
for word in word_list:
    index_in_dictionary = dictionary[word]
    word_embedding = embeddings[index_in_dictionary]
    #print(word_embedding)
    word_embedding_array = np.array(word_embedding)
    #print(word_embedding_array.shape)
    
    #Define a dictionary to store the similarities.
    sim = {}
    
    #Now we traverse the dictionary to calculate the similarity between the word in our 3-word list v/s the dictionary of words.
    for dict_word in dictionary.keys():
        #we take only those words which are not equal to our target words, as the similarity for those would be one!
        if dict_word != word:
            dict_index = dictionary[dict_word]
            embed_word = np.array(embeddings[dict_index])
            #The cosine similarity is calculated as the dot product divide by the normalized values.
            cos_sim = dot(embed_word,word_embedding_array)/(norm(embed_word)*norm(word_embedding_array))  
           #cos_sim = dot(embed_word,word_embedding_array)
  
            sim[dict_word] = cos_sim     
        
    #Now we reverse sort the dictionary as per the cosine similarity values.
    list_items = sim.items()
    reverse_sim = sorted(list_items,key=operator.itemgetter(1),reverse = True)
    #print(reverse_sim)
    top_20 = [key for key,value in reverse_sim[:20]]
    print(word)
    print(top_20)                                                             
    #result2 = "\" ".join("\""+word+"\"")
    #result2 = result2+(["\""+k for k in top_2])+"\""
    #word_predictions.write(result2 + '\n')
#print(word_embeddings)
            
output_file.close()