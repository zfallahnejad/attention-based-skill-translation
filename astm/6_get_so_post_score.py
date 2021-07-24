import os
import re
import pickle
import statistics

so_output_path = "./data/"
if not os.path.exists(so_output_path):
    os.makedirs(so_output_path)

for target_tag, so_input_path in [("java", "./data/JavaPosts.xml"), ("php", "./data/PhpPosts.xml")]:
    post_score = {}
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    score_regex = re.compile("(?<=Score=\")(?P<Score>.*?)(?=\" )")
    with open(so_output_path + "so_{}_post_score.txt".format(target_tag), "w") as outfile:
        with open(so_input_path, encoding='utf8') as java_posts_file:
            for line in java_posts_file:
                post_id = id_regex.search(line).group('Id')
                score = int(score_regex.search(line).group('Score'))
                outfile.write("{},{}\n".format(post_id, score))
                post_score[post_id] = score

    with open(os.path.join(so_output_path, "so_{}_post_score.pkl".format(target_tag)), 'wb') as output:
        pickle.dump(post_score, output)

    scores = [post_score[pid] for pid in post_score]
    score_freq = {l: scores.count(l) for l in set(scores)}
    print("Average score=", statistics.mean(scores))
    print("Median score=", statistics.median(scores))
    print("Minimum score={}, number of posts={}".format(min(scores), score_freq[min(scores)]))
    print("Maximum score={}, number of posts={}".format(max(scores), score_freq[max(scores)]))
    max_count = max(score_freq.values())
    length_max_count = list(score_freq.keys())[list(score_freq.values()).index(max_count)]
    print("Maximum frequency={}, with score={}".format(max_count, length_max_count))

'''
tag: java
Average score= 1.9745674383413554
Median score= 1
Minimum score=-40, number of posts=1
Maximum score=14639, number of posts=1
Maximum frequency=944902, with score=0
'''
'''
tag: php
Average score= 1.2610038940008934
Median score= 0
Minimum score=-147, number of posts=1
Maximum score=3610, number of posts=1
Maximum frequency=953720, with score=0
'''
