import argparse
import numpy as np
from gensim.models import KeyedVectors


def main(tag, size, window, negative, min_count, data_path):
    word_vectors_path = "{}-wv-d{}-win{}-neg{}-min{}.kv".format(tag, size, window, negative, min_count)
    word_vectors = KeyedVectors.load(word_vectors_path, mmap='r')
    print('We trained %s word vectors using gensim word2vec module.' % len(word_vectors.vectors))

    original_post_len = []
    preprocessed_post_len = []
    with open(data_path, encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split()
            # post_id = line[0]
            original_post_len.append(len(line) - 1)

            body = []
            for word in line[1:]:
                if word in word_vectors:
                    body.append(word)
            preprocessed_post_len.append(len(body))

    print("minimum length of post:\toriginal:{}\tpreprocessed:{}".format(np.min(original_post_len),
                                                                         np.min(preprocessed_post_len)))
    print("maximum length of post:\toriginal:{}\tpreprocessed:{}".format(np.max(original_post_len),
                                                                         np.max(preprocessed_post_len)))
    print("mean of post length:\toriginal:{}\tpreprocessed:{}".format(int(np.mean(original_post_len)),
                                                                      int(np.mean(preprocessed_post_len))))
    print("median of post length:\toriginal:{}\tpreprocessed:{}".format(np.median(original_post_len),
                                                                        np.median(preprocessed_post_len)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help="", required=True)
    parser.add_argument('--size', type=int, help="", required=True)
    parser.add_argument('--window', type=int, help="", required=True)
    parser.add_argument('--negative', type=int, help="", required=True)
    parser.add_argument('--min_count', type=int, help="", required=True)
    parser.add_argument('--data_path', help='', required=True)
    args = parser.parse_args()

    main(args.tag, args.size, args.window, args.negative, args.min_count, args.data_path)

'''
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
'''
