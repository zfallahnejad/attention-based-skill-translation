import os
import argparse
from gensim.models import Word2Vec


class SentenceIterator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.words_freq = {}
        self.first_time = True

    def __iter__(self):
        print("first_time:", self.first_time)
        for line in open(self.filepath, encoding="utf8"):
            line = line.strip().split()
            if self.first_time:
                for t in line[1:]:
                    if t in self.words_freq:
                        self.words_freq[t] += 1
                    else:
                        self.words_freq[t] = 1
            yield line[1:]
        self.first_time = False


def main(
        tag,
        size=100,
        window=5,
        negative=5,
        min_count=1,
        n_workers=20,
        nr_iter=5):
    result_path = "./word_vectors/word2vec_model/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    sentences = SentenceIterator('./data/so_{}.txt'.format(tag))
    print("Start training..")
    model = Word2Vec(
        sentences=sentences,
        size=size,  # Dimension of the word vectors
        window=window,  # Context window size
        min_count=min_count,  # Min count
        workers=n_workers,  # Number of workers
        sample=1e-5,
        negative=negative,  # Number of negative samples (usually between 5-20)
        iter=nr_iter,  # Number of iterations
    )
    # model.save("./word2vec_model/fa-wv-d" + str(size) + "-win" + str(window) + "-neg" + str(negative) + ".model")
    # print(os.listdir("./word2vec_model"))
    model.wv.save(result_path + "{}-wv-d{}-win{}-neg{}-min{}.kv".format(tag, size, window, negative, min_count))
    print("Number of word vectors:", len(model.wv.vectors))  # 16511

    print("Number of unique words in this dataset: ", len(sentences.words_freq))  # 80023
    print("Number of words with freq equal or above {}: ".format(min_count),
          len([w for w in sentences.words_freq if sentences.words_freq[w] >= min_count]))  # 12577


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help="", required=False)
    parser.add_argument('--size', type=int, help="", required=False)
    parser.add_argument('--window', type=int, help="", required=False)
    parser.add_argument('--negative', type=int, help="", required=False)
    parser.add_argument('--min_count', type=int, help="", required=False)
    args = parser.parse_args()

    main(args.tag, args.size, args.window, args.negative, args.min_count)

'''
python 4_train_word_vectors.py --tag java --size 100 --window 5 --negative 5 --min_count 3
'''
