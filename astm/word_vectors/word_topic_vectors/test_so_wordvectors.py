import argparse
import numpy as np


def main(tag, embed_size, data_path):
    words = set()
    with open("word_topic_vectors_{}_{}d.txt".format(tag, embed_size), encoding="utf8") as vector_file:
        for line in vector_file:
            line = line.strip().split('\t')
            word = line[0]
            words.add(word)
    print('We trained %s word vectors using lda word2vec module.' % len(words))

    original_post_len = []
    preprocessed_post_len = []
    with open(data_path, encoding="utf8") as infile:
        for line in infile:
            line = line.strip().split()
            # post_id = line[0]
            original_post_len.append(len(line) - 1)

            body = []
            for word in line[1:]:
                if word in words:
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
    parser.add_argument('--embed_size', type=int, help="", required=True)
    parser.add_argument('--data_path', help='', required=True)
    args = parser.parse_args()

    main(args.tag, args.embed_size, args.data_path)

'''
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
'''
