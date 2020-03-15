================================================================================================================
	
		Environment details: 
		Python              : 2.7.16
		TensorFlow          : 1.5.0

		Best model for cross entropy loss:
		batch_size          : 192
		skip_window         : 8
		num_skips           : 8
		max_no_of_steps		: 200001
		num_sampled         : 64
		learning_rate       : 1.0


		Best model for noise contrastive estimation loss:
		batch_size          : 192 
		skip_window         : 4
		num_skips           : 8
		max_no_of_steps		: 100001
		num_sampled         : 64
		learning_rate       : 1.0

==================================================================================================================
1. First we begin with batch generation.
   Function name: generate_batch
   Parameters passed: data, batch_size, num_skips, skip_window
   Function signature: generate_batch(data, batch_size, num_skips, skip_window)
   File name: word2vec_basic.py
   Parameters passed: 
   @data_index: the index of a word. A word can be accessed by using data[data_index]
   @batch_size: the number of instances in one batch
   @num_skips: the number of samples you want to draw in a window 
            (In the below example, it was 2)
   @skip_windows: decides how many words to consider left and right from a context word. 
                (So, skip_windows*2+1 = window_size)
				
   batch will contain word ids for context words. Dimension is [batch_size].
   labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].


   I followed the below approach for generating batch:
    - To generate batches, I have used a naive approach and i traversed the dataset one word at a time adding it to the list.
	- Once we have the required number of samples based on the current window size, I increment the data_index value.
   	
	
2. Implement cross entropy loss
   Function name       : cross_entropy_loss
   File name  		   : loss_func.py
   
   Parameters
   inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
   true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
   
   - To calculate A, I have used matrix multiplication of tensorflow. This returns me the matrix multiplication between the transpose of context words and the target words.
   - Since the embeddings for target words are basically one hot vectors, once we multiply it with T(inputs), the resultant matrix has our desired values as diagonal elements. 
   - The diagonal matrix of above matmul is my first value - A = {u_o}^T v_c
   
   - To calculate B, I have first taken exponent of matrix multiplication we calculated for A.
   - Followed by the sum of above term and then the log of it.
   - This gives me B = B = log(\sum{exp({u_w}^T v_c)})

   Link to model:   https://drive.google.com/file/d/1dfjgz-0jAkmfcEuNRirZk5j8RIMXF_yF/view?usp=sharing
   
3. Implement NCE loss function.
   Function name      :  nce_loss
   File name		  :  loss_func.py
   
   I have explained NCE with comments in my code. Please refer to the code in loss_func.py

   Link to model: https://drive.google.com/file/d/1950A203s9wxGeJxmlCSPZW0VnnB_u4Ul/view?usp=sharing
   
4. Word analogies using word vectors
   File name      : word_analogy.py
   - To find the word analogies, I have made use of the method described in the pdf document provided:
     Recall that vectors are representing some direction in space. 
	 If (a, b) and (c, d) pairs are analogous pairs then the transformation from a to b (i.e., some x vector when added to a gives b: a + x = b) 
     should be highly similar to the transformation from c to d (i.e., some y vector when added to c gives d: b + y = d). 
	 In other words, the difference vector (b-a) should be similar to difference vector (d-c). 
	 This difference vector can be thought to represent the relation between the two words. 
  
	 To implement above, I have used the following approach.
   - I have initially split my entire input file into left and right vectors based on the pipe symbol. 
   - Furthermore each left and right parts are further split based on ','. This gives us the word_pairs.
   - I have then split these word pairs into individual words based ":".
   - I have then lookedup embeddings for each of these words into embeddings by embeddings[dictionary[word]]
   - Now that we have the word embeddings, I have calculated the difference vectors for each pair where if a:b then embedding(b) - embedding(a).
   - For the left vectors, I have then used np.mean to calculate the average of the vectors.
   - So now, for every row in the file, we have one mean value on the left, with 3 difference values on the right.
   - Now for every word in left vector, I have calculated the cosine similarity with the words in right vector.
   - Once done, I have written the minimum values and maximum values to the output file(indicating least illustrative pair and most illustrative pair)
   
5. Finding the top 20 similar words to {'first','american','would'}
    File name   : word_analogy.py
   - To find the top 20 similar words, I have done the following:
     - for every word wt in our word list of 3, I did below:
		- for every word w1 in dictionary other than above three, I stored their embeddings by embedding lookup.
		- Calculated cosine similarity between each w1 and the wt and stored each of the similarities into a dictionary with key as w1 words and values as their similairty with wt.
		- Reverse sorted the dictionary by values and returned the top 20 for each word wt.
		
