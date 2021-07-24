import os
import pickle
import argparse


def build_word_topic_vectors(tag, embed_size):
    word_index = {}
    index_word = {}
    input_file = "./mallet-2.0.8/output/word-topic-counts-file-{}-{}.txt".format(tag, embed_size)

    result_path = "./word_vectors/word_topic_vectors"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    output_file = open(os.path.join(result_path, "word_topic_vectors_{}_{}d.txt".format(tag, embed_size)), "w",
                       encoding="utf8")

    with open(input_file, encoding='utf8') as posts_file:
        for line in posts_file:
            print(line.strip())
            line = line.strip().split(" ")
            index = int(line.pop(0))
            word = line.pop(0)

            topic_count = {}
            sum_count = 0
            for tc in line:
                topic, count = tc.split(":")
                topic_count[int(topic)] = int(count)
                sum_count += int(count)

            out_str = word + "\t"
            for topic in range(embed_size):
                if topic in topic_count:
                    out_str += '{0:.5f} '.format(topic_count[topic] / sum_count)
                else:
                    out_str += "0 "
            output_file.write(out_str[:-1] + "\n")

            word_index[word] = index
            index_word[index] = word
    output_file.close()

    with open(os.path.join(result_path, "word_index_{}_{}.pkl".format(tag, embed_size)), 'wb') as output:
        pickle.dump(word_index, output)

    with open(os.path.join(result_path, "index_word_{}_{}.pkl".format(tag, embed_size)), 'wb') as output:
        pickle.dump(index_word, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help="", required=True)
    parser.add_argument('--embed_size', type=int, help="", required=True)
    args = parser.parse_args()

    build_word_topic_vectors(args.tag, args.embed_size)
