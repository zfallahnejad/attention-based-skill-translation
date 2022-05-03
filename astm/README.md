# Download Posts.xml
Here is the [link](https://drive.google.com/file/d/1OPBCLwgTCkzCzdpN_F4O5TuSCPtA3IdP/view?usp=sharing) to our dataset.
Download this 7-Gigabyte 7z file which contains StackOverflow `Posts.xml` file and extract it inside `data` directory.
 
# 0.0. Extract a sub collection of Stackoverflow posts based on a single tag
For a single tag like `java`, we extract questions of stackoverflow which tagged by `java` along their answers.
```
python3 0_0_so_sub_collection.py

number of java questions: 810071
number of java answers: 1510812

number of php questions: 714476
number of php answers: 1298107
```

# 0.1 Add tags to each answer
For each answer, we take the tags of its parent and append it to the data of that answer, So that the new xml file
contains tags for questions and answers.
```
python3 0_1_add_parent_tags_to_answers.py
```

# 0.2 Golden set
```
cd golden_set
python3 golden_creator.py
```
`golden` directory inside `golden_set` which is the result of running `golden_creator.py` scripts.
In this script we try to re-generate golden set of `Quality-aware skill translation models for expert finding on
StackOverflow` paper which is available in [this repository](https://github.com/arashdn/sof-expert-finding).
`neshati` directory inside `golden_set` contains the golden set of this paper.
By change the threshold for php dataset from 8(reported by this paper) to 7, we managed to reproduce their golden set.
But no matter how hard we tried, we could not produce a similar Java golden set, and the produced golden set was slightly different from theirs.
We also tried to reproduce this golden set with a Java code, but again the set produced was different from them.     

# 1. prepare stack overflow dataset
This script load your desired sub collection of stack overflow dataset, normalize its text and prepare 
another file (for example `so_java.txt` for java collection) to be used as the input of other scripts.
```
python3 1_prepare_so.py
```
Note: First post id of stackoverflow is `123` and last one is `28922954`.
```
Stackoverflow dataset:
number of posts:  24120526
number of tags:  39837

Java subeset of stackoverflow dataset:
number of posts:  2320883
number of tags:  18957
```

# 2. Train LDA
```
tar -zxvf mallet-2.0.8.tar.gz
mkdir ./mallet-2.0.8/output
```
Before start training, you should change value of `MEMORY=1g` inside `./mallet-2.0.8/bin/mallet` file to `MEMORY=32g`.
Then, run the following script
```
sh 2_train_lda.sh
```

# 3. Build word topic vectors
We will use output of previously trained lda to build vectors for each words.
As we can see, we get `7194757` word vectors from lda outputs of java collection and  `4498999` word vectors 
from lda outputs of php collection.
```
python3 3_build_word_topic_vectors.py --tag java --embed_size 50
python3 3_build_word_topic_vectors.py --tag java --embed_size 100
python3 3_build_word_topic_vectors.py --tag java --embed_size 150
python3 3_build_word_topic_vectors.py --tag java --embed_size 200
python3 3_build_word_topic_vectors.py --tag php --embed_size 50
python3 3_build_word_topic_vectors.py --tag php --embed_size 100
python3 3_build_word_topic_vectors.py --tag php --embed_size 150
python3 3_build_word_topic_vectors.py --tag php --embed_size 200
python3 3_build_word_topic_vectors.py --tag android --embed_size 50
python3 3_build_word_topic_vectors.py --tag android --embed_size 100
python3 3_build_word_topic_vectors.py --tag android --embed_size 150
python3 3_build_word_topic_vectors.py --tag android --embed_size 200
python3 3_build_word_topic_vectors.py --tag c# --embed_size 50
python3 3_build_word_topic_vectors.py --tag c# --embed_size 100
python3 3_build_word_topic_vectors.py --tag c# --embed_size 150
python3 3_build_word_topic_vectors.py --tag c# --embed_size 200
```
After training word vectors, I ran `test_so_wordvectors.py` script to know some statistics about document length.
```
python3 test_so_wordvectors.py --tag java --embed_size 50 --data_path ../../data/so_java.txt
We trained 7194758 word vectors using lda word2vec module.
minimum length of post: original:2      preprocessed:0
maximum length of post: original:7681   preprocessed:1963
mean of post length:    original:130    preprocessed:39
median of post length:  original:90.0   preprocessed:27.0

python3 test_so_wordvectors.py --tag java --embed_size 100 --data_path ../../data/so_java.txt
We trained 7194758 word vectors using lda word2vec module.
minimum length of post: original:2      preprocessed:0
maximum length of post: original:7681   preprocessed:1963
mean of post length:    original:130    preprocessed:39
median of post length:  original:90.0   preprocessed:27.0

python3 test_so_wordvectors.py --tag java --embed_size 150 --data_path ../../data/so_java.txt
We trained 7194758 word vectors using lda word2vec module.
minimum length of post: original:2      preprocessed:0
maximum length of post: original:7681   preprocessed:1963
mean of post length:    original:130    preprocessed:39
median of post length:  original:90.0   preprocessed:27.0

python3 test_so_wordvectors.py --tag java --embed_size 200 --data_path ../../data/so_java.txt
We trained 7194758 word vectors using lda word2vec module.
minimum length of post: original:2      preprocessed:0
maximum length of post: original:7681   preprocessed:1963
mean of post length:    original:130    preprocessed:39
median of post length:  original:90.0   preprocessed:27.0

python3 test_so_wordvectors.py --tag php --embed_size 50 --data_path ../../data/so_php.txt
We trained 4499000 word vectors using lda word2vec module.
minimum length of post: original:3      preprocessed:0
maximum length of post: original:5714   preprocessed:2487
mean of post length:    original:117    preprocessed:29
median of post length:  original:83.0   preprocessed:21.0

python3 test_so_wordvectors.py --tag php --embed_size 100 --data_path ../../data/so_php.txt
We trained 4499000 word vectors using lda word2vec module.
minimum length of post: original:3      preprocessed:0
maximum length of post: original:5714   preprocessed:2487
mean of post length:    original:117    preprocessed:29
median of post length:  original:83.0   preprocessed:21.0

python3 test_so_wordvectors.py --tag php --embed_size 150 --data_path ../../data/so_php.txt
We trained 4499000 word vectors using lda word2vec module.
minimum length of post: original:3      preprocessed:0
maximum length of post: original:5714   preprocessed:2487
mean of post length:    original:117    preprocessed:29
median of post length:  original:83.0   preprocessed:21.0

python3 test_so_wordvectors.py --tag php --embed_size 200 --data_path ../../data/so_php.txt
We trained 4499000 word vectors using lda word2vec module.
minimum length of post: original:3      preprocessed:0
maximum length of post: original:5714   preprocessed:2487
mean of post length:    original:117    preprocessed:29
median of post length:  original:83.0   preprocessed:21.0
```
Based on these results, it seems that I can test `64`, `128`, `256` or `512` as the value of maximum sentence length of my code. 

# 4. Train word vectors using gensim word2vec
Note: The final results of astm-1 and astm-2 are based on topic vectors.
The results of our experiments showed that the results of the models using topic vectors were better than the results of the models using word2vec vectors! 
I trained word2vec so many times with so many normalization step but I did not get better results from them. 
So, you can skip this section if you want to reproduce our final results. 
```
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 1
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 2
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 3
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 5
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag java --size 50 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 5 --min_count 3
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 5 --min_count 5
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 10 --min_count 3
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 10 --min_count 5
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 10 --min_count 7
python3 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 10 --min_count 10
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 5 --min_count 3
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 5 --min_count 5
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 10 --min_count 3
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 10 --min_count 5
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 10 --min_count 7
python3 4_train_word_vectors.py --tag java --size 200 --window 5 --negative 10 --min_count 10
```
After training word vectors, I ran `test_so_wordvectors.py` script to know some statistics about document length.
```
python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 1 --data_path ../../data/so_java.txt
We trained 12423935 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:2
maximum length of post: original:7681   preprocessed:7681
mean of post length:    original:130    preprocessed:130
median of post length:  original:90.0   preprocessed:90.0

python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 2 --data_path ../../data/so_java.txt
We trained 3819671 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:1
maximum length of post: original:7681   preprocessed:6361
mean of post length:    original:130    preprocessed:127
median of post length:  original:90.0   preprocessed:88.0

python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 3 --data_path ../../data/so_java.txt
We trained 2008571 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:1
maximum length of post: original:7681   preprocessed:6225
mean of post length:    original:130    preprocessed:125
median of post length:  original:90.0   preprocessed:87.0

python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 5 --data_path ../../data/so_java.txt
We trained 1022986 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:1
maximum length of post: original:7681   preprocessed:5763
mean of post length:    original:130    preprocessed:124
median of post length:  original:90.0   preprocessed:87.0

python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 7 --data_path ../../data/so_java.txt
We trained 695904 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:1
maximum length of post: original:7681   preprocessed:5729
mean of post length:    original:130    preprocessed:123
median of post length:  original:90.0   preprocessed:86.0

python3 test_so_wordvectors.py --tag java --size 50 --window 5 --negative 5 --min_count 10 --data_path ../../data/so_java.txt
We trained 479413 word vectors using gensim word2vec module.
minimum length of post: original:2      preprocessed:1
maximum length of post: original:7681   preprocessed:5715
mean of post length:    original:130    preprocessed:122
median of post length:  original:90.0   preprocessed:86.0
```
Based on these results, it seems that I must test `128`, `256` or `512` as the value of maximum sentence length. 

And for php collection:
```
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 1
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 2
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 3
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 5
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag php --size 50 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag php --size 100 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag php --size 100 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag php --size 100 --window 5 --negative 10 --min_count 7
python3 4_train_word_vectors.py --tag php --size 100 --window 5 --negative 10 --min_count 10
python3 4_train_word_vectors.py --tag php --size 200 --window 5 --negative 5 --min_count 7
python3 4_train_word_vectors.py --tag php --size 200 --window 5 --negative 5 --min_count 10
python3 4_train_word_vectors.py --tag php --size 200 --window 5 --negative 10 --min_count 7
python3 4_train_word_vectors.py --tag php --size 200 --window 5 --negative 10 --min_count 10
```
After training word vectors, I ran `test_so_wordvectors.py` script to know some statistics about document length.
```
python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 1 --data_path ../../data/so_php.txt
We trained 10201275 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:3
maximum length of post: original:5714   preprocessed:5714
mean of post length:    original:117    preprocessed:117
median of post length:  original:83.0   preprocessed:83.0

python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 2 --data_path ../../data/so_php.txt
We trained 3179589 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:1
maximum length of post: original:5714   preprocessed:5668
mean of post length:    original:117    preprocessed:113
median of post length:  original:83.0   preprocessed:81.0

python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 3 --data_path ../../data/so_php.txt
We trained 1655854 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:1
maximum length of post: original:5714   preprocessed:5666
mean of post length:    original:117    preprocessed:112
median of post length:  original:83.0   preprocessed:80.0

python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 5 --data_path ../../data/so_php.txt
We trained 822543 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:1
maximum length of post: original:5714   preprocessed:5666
mean of post length:    original:117    preprocessed:110
median of post length:  original:83.0   preprocessed:79.0

python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 7 --data_path ../../data/so_php.txt
We trained 550540 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:1
maximum length of post: original:5714   preprocessed:5666
mean of post length:    original:117    preprocessed:110
median of post length:  original:83.0   preprocessed:79.0

python3 test_so_wordvectors.py --tag php --size 50 --window 5 --negative 5 --min_count 10 --data_path ../../data/so_php.txt
We trained 375436 word vectors using gensim word2vec module.
minimum length of post: original:3      preprocessed:1
maximum length of post: original:5714   preprocessed:5663
mean of post length:    original:117    preprocessed:109
median of post length:  original:83.0   preprocessed:79.0
```
Based on these results, it seems that I must test `128`, `256` or `512` as the value of maximum sentence length. 

# 5. Extract tags of posts
This script will generate a dictionary of post_id and their tags, and save this dictionary inside 
`./data/so_java_post_tags.pkl` for java collection.
```
python3 5_get_so_post_tags.py
```
It should be noted that, we just kept top 100 tags and removed other tags even 'java'. After that all the post without any tag was ignored.
We build three version of this dictionary. One for all of java posts and the others for question and answers only. Same steps for php.

# 6. Extract score of posts
This script will write a file in which each line contains post id and score of that post
```
python3 6_get_so_post_score.py
```

# 7. Analyze users number of posts
This script will write a file in which each line contains post id and score of that post
```
python3 7_analyze_user_num_posts.py
```

# Multi-Label classification (stackoverflow)


