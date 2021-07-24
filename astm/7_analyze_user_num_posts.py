import os
import re

so_output_path = "./data/"
if not os.path.exists(so_output_path):
    os.makedirs(so_output_path)

for tag, so_input_path in [("java", "./data/JavaPosts.xml"), ("php", "./data/PhpPosts.xml")]:
    user_post = {}
    post_score = {}
    post_type = {}
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    score_regex = re.compile("(?<=Score=\")(?P<Score>.*?)(?=\" )")
    owner_regex = re.compile("(?<=OwnerUserId=\")(?P<OwnerUserId>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    with open(so_input_path, encoding='utf8') as posts_file:
        for line in posts_file:
            try:
                post_id = id_regex.search(line).group('Id')
                owner_id = int(owner_regex.search(line).group('OwnerUserId'))
                score = int(score_regex.search(line).group('Score'))
                post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))

                if owner_id == -1:
                    pass

                if owner_id in user_post:
                    user_post[owner_id].append(post_id)
                else:
                    user_post[owner_id] = [post_id]
                post_score[post_id] = score
                post_type[post_id] = post_type_id
                print(post_id, owner_id, score)
            except:
                pass

    print("number of users: ", len(user_post))
    print("number of posts: ", len(post_score))
    num_users_with_posts_score_gt_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0, 25: 0}
    num_users_with_questions_score_gt_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0,
                                            25: 0}
    num_users_with_answers_score_gt_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0,
                                          25: 0}

    max_num_posts = 1
    max_num_posts_score_ge_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0, 25: 0}
    max_num_questions_score_ge_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0, 25: 0}
    max_num_answers_score_ge_th = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 10: 0, 12: 0, 13: 0, 15: 0, 20: 0, 25: 0}

    for user in user_post:
        max_num_posts = max(max_num_posts, len(user_post[user]))
        for th in max_num_posts_score_ge_th:
            posts_with_score_ge_th = [p for p in user_post[user] if post_score[p] >= th]
            questions_with_score_ge_th = [p for p in user_post[user] if post_type[p] == 1 and post_score[p] >= th]
            answers_with_score_ge_th = [p for p in user_post[user] if post_type[p] == 2 and post_score[p] >= th]

            if posts_with_score_ge_th:
                num_users_with_posts_score_gt_th[th] += 1
            if questions_with_score_ge_th:
                num_users_with_questions_score_gt_th[th] += 1
            if answers_with_score_ge_th:
                num_users_with_answers_score_gt_th[th] += 1

            max_num_posts_score_ge_th[th] = max(max_num_posts_score_ge_th[th], len(posts_with_score_ge_th))
            max_num_questions_score_ge_th[th] = max(max_num_questions_score_ge_th[th], len(questions_with_score_ge_th))
            max_num_answers_score_ge_th[th] = max(max_num_answers_score_ge_th[th], len(answers_with_score_ge_th))

    print("maximum number of post each user have: ", max_num_posts)
    for th in [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 15, 20, 25]:
        print("We have {} users with posts with score >= {} and maximum number of these user's posts are: {}".format(
            num_users_with_posts_score_gt_th[th], th, max_num_posts_score_ge_th[th]))
    for th in [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 15, 20, 25]:
        print(
            "We have {} users with questions with score >= {} and maximum number of these user's questions are: {}".format(
                num_users_with_questions_score_gt_th[th], th, max_num_questions_score_ge_th[th]))
    for th in [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 15, 20, 25]:
        print("We have {} users with answers with score >= {} and maximum number of these user's answers are: {}".format(
            num_users_with_answers_score_gt_th[th], th, max_num_answers_score_ge_th[th]))

'''
tag: java
number of users:  394061
number of posts:  2303918
maximum number of post each user have:  10770

We have 374291 users with posts with score >= 0 and maximum number of these user's posts are: 10678
We have 237515 users with posts with score >= 1 and maximum number of these user's posts are: 9039
We have 148958 users with posts with score >= 2 and maximum number of these user's posts are: 7940
We have 100020 users with posts with score >= 3 and maximum number of these user's posts are: 6667
We have 72216 users with posts with score >= 4 and maximum number of these user's posts are: 5448
We have 55808 users with posts with score >= 5 and maximum number of these user's posts are: 4393
We have 37670 users with posts with score >= 7 and maximum number of these user's posts are: 2854
We have 25121 users with posts with score >= 10 and maximum number of these user's posts are: 1629
We have 20542 users with posts with score >= 12 and maximum number of these user's posts are: 1196
We have 18788 users with posts with score >= 13 and maximum number of these user's posts are: 1052

We have 16100 users with posts with score >= 15 and maximum number of these user's posts are: 849
We have 11689 users with posts with score >= 20 and maximum number of these user's posts are: 552
We have 9062 users with posts with score >= 25 and maximum number of these user's posts are: 420

We have 253032 users with questions with score >= 0 and maximum number of these user's questions are: 818

We have 151878 users with questions with score >= 1 and maximum number of these user's questions are: 523
We have 92109 users with questions with score >= 2 and maximum number of these user's questions are: 334
We have 59739 users with questions with score >= 3 and maximum number of these user's questions are: 221
We have 41921 users with questions with score >= 4 and maximum number of these user's questions are: 158
We have 31524 users with questions with score >= 5 and maximum number of these user's questions are: 122
We have 20381 users with questions with score >= 7 and maximum number of these user's questions are: 90
We have 13010 users with questions with score >= 10 and maximum number of these user's questions are: 62
We have 10467 users with questions with score >= 12 and maximum number of these user's questions are: 51
We have 9497 users with questions with score >= 13 and maximum number of these user's questions are: 48
We have 7988 users with questions with score >= 15 and maximum number of these user's questions are: 46
We have 5580 users with questions with score >= 20 and maximum number of these user's questions are: 40
We have 4221 users with questions with score >= 25 and maximum number of these user's questions are: 34

We have 201888 users with answers with score >= 0 and maximum number of these user's answers are: 10654
We have 128505 users with answers with score >= 1 and maximum number of these user's answers are: 9039
We have 81466 users with answers with score >= 2 and maximum number of these user's answers are: 7940
We have 56146 users with answers with score >= 3 and maximum number of these user's answers are: 6667
We have 41433 users with answers with score >= 4 and maximum number of these user's answers are: 5448
We have 32562 users with answers with score >= 5 and maximum number of these user's answers are: 4393
We have 22496 users with answers with score >= 7 and maximum number of these user's answers are: 2854
We have 15290 users with answers with score >= 10 and maximum number of these user's answers are: 1629
We have 12530 users with answers with score >= 12 and maximum number of these user's answers are: 1196
We have 11487 users with answers with score >= 13 and maximum number of these user's answers are: 1052

We have 9922 users with answers with score >= 15 and maximum number of these user's answers are: 849
We have 7286 users with answers with score >= 20 and maximum number of these user's answers are: 552
We have 5694 users with answers with score >= 25 and maximum number of these user's answers are: 420
'''
'''
tag: php
number of users:  350961
number of posts:  1994289
maximum number of post each user have:  6661

We have 332121 users with posts with score >= 0 and maximum number of these user's posts are: 6587
We have 195745 users with posts with score >= 1 and maximum number of these user's posts are: 4554
We have 112597 users with posts with score >= 2 and maximum number of these user's posts are: 2650
We have 69663 users with posts with score >= 3 and maximum number of these user's posts are: 1707
We have 47835 users with posts with score >= 4 and maximum number of these user's posts are: 1118
We have 35323 users with posts with score >= 5 and maximum number of these user's posts are: 813
We have 22481 users with posts with score >= 7 and maximum number of these user's posts are: 490
We have 14202 users with posts with score >= 10 and maximum number of these user's posts are: 305
We have 11239 users with posts with score >= 12 and maximum number of these user's posts are: 248
We have 10216 users with posts with score >= 13 and maximum number of these user's posts are: 211
We have 8564 users with posts with score >= 15 and maximum number of these user's posts are: 166
We have 6023 users with posts with score >= 20 and maximum number of these user's posts are: 120
We have 4524 users with posts with score >= 25 and maximum number of these user's posts are: 87

We have 214327 users with questions with score >= 0 and maximum number of these user's questions are: 444
We have 113484 users with questions with score >= 1 and maximum number of these user's questions are: 299
We have 60133 users with questions with score >= 2 and maximum number of these user's questions are: 204
We have 34722 users with questions with score >= 3 and maximum number of these user's questions are: 156
We have 22575 users with questions with score >= 4 and maximum number of these user's questions are: 112
We have 16051 users with questions with score >= 5 and maximum number of these user's questions are: 81
We have 9718 users with questions with score >= 7 and maximum number of these user's questions are: 54
We have 5838 users with questions with score >= 10 and maximum number of these user's questions are: 37
We have 4535 users with questions with score >= 12 and maximum number of these user's questions are: 35
We have 4035 users with questions with score >= 13 and maximum number of these user's questions are: 30
We have 3292 users with questions with score >= 15 and maximum number of these user's questions are: 28
We have 2202 users with questions with score >= 20 and maximum number of these user's questions are: 21
We have 1638 users with questions with score >= 25 and maximum number of these user's questions are: 15

We have 186860 users with answers with score >= 0 and maximum number of these user's answers are: 6587
We have 113960 users with answers with score >= 1 and maximum number of these user's answers are: 4554
We have 68497 users with answers with score >= 2 and maximum number of these user's answers are: 2650
We have 44320 users with answers with score >= 3 and maximum number of these user's answers are: 1697
We have 31254 users with answers with score >= 4 and maximum number of these user's answers are: 1110
We have 23544 users with answers with score >= 5 and maximum number of these user's answers are: 768
We have 15219 users with answers with score >= 7 and maximum number of these user's answers are: 482
We have 9745 users with answers with score >= 10 and maximum number of these user's answers are: 305
We have 7710 users with answers with score >= 12 and maximum number of these user's answers are: 248
We have 7054 users with answers with score >= 13 and maximum number of these user's answers are: 211
We have 5961 users with answers with score >= 15 and maximum number of these user's answers are: 166
We have 4239 users with answers with score >= 20 and maximum number of these user's answers are: 116
We have 3178 users with answers with score >= 25 and maximum number of these user's answers are: 87
'''
