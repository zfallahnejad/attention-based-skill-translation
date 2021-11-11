import re


def load_file(file_name):
    pid_scores = {}
    with open(file_name, encoding="utf8") as infile:
        for line in infile:
            pid, score = [int(_) for _ in line.strip().split(',')]
            pid_scores[pid] = score
    return pid_scores


def load_voteshare_file(file_name):
    pid_scores = {}
    with open(file_name, encoding="utf8") as infile:
        for line in infile:
            if line.strip() == "aid,voteshare":
                continue
            pid, score = [float(_) for _ in line.strip().split(',')]
            pid_scores[pid] = score
    return pid_scores


def load_post_xml(xml_input):
    question_list, answer_list = [], []
    answer_parent, answer_score, question_answers = {}, {}, {}
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    parent_id_regex = re.compile("(?<=ParentId=\")(?P<ParentId>.*?)(?=\" )")
    score_regex = re.compile("(?<=Score=\")(?P<Score>.*?)(?=\" )")
    with open(xml_input, encoding='utf8') as posts_file:
        for line in posts_file:
            if not line.strip().startswith('<row'):
                continue
            post_id = int(id_regex.search(line).group('Id'))
            post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
            if post_type_id == 2:
                # answers
                answer_list.append(post_id)
                parent_id = int(parent_id_regex.search(line).group('ParentId'))
                score = int(score_regex.search(line).group('Score'))
                answer_parent[post_id] = parent_id
                answer_score[post_id] = score
                if parent_id in question_answers:
                    question_answers[parent_id].append(post_id)
                else:
                    question_answers[parent_id] = [post_id]
                    question_list.append(parent_id)
    return question_list, answer_list, answer_parent, answer_score, question_answers


def case_1(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    The vote-share of a document with a negative vote-score is (probably) negative, which is possibly harmless,
    depending on how a model handles the score, but counter-intuitive, given the name of the measure.
    """
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    with open('./logs/{}_case1.txt'.format(tag), 'w', encoding="utf8") as outfile:
        pos_voteshare, zero_voteshare, neg_voteshare = 0, 0, 0
        num_thread = 0
        for qid in question_list:
            for aid in question_answers.get(qid):
                if post_scores[aid] < 0:
                    num_thread += 1
                    print('qid:{}, aid:{}, score:{}, pre_computed_voteshare:{}'.format(
                        qid, aid, post_scores[aid], aid_voteshares.get(aid, default_voteshare))
                    )
                    outfile.write('qid:{}, aid:{}, score:{}, pre_computed_voteshare:{}\n'.format(
                        qid, aid, post_scores[aid], aid_voteshares.get(aid, default_voteshare))
                    )
                    if aid_voteshares.get(aid, default_voteshare) > 0:
                        pos_voteshare += 1
                    elif aid_voteshares.get(aid, default_voteshare) == 0:
                        zero_voteshare += 1
                    else:
                        neg_voteshare += 1
        print("number of questions or threads with at least one answer: {}".format(len(question_list)))
        outfile.write("number of questions or threads with at least one answer: {}\n".format(len(question_list)))
        print("number of answers with negative scores: {}".format(num_thread))
        outfile.write("number of answers with negative scores: {}\n".format(num_thread))
        print("#answers with negative score and positive precomputed voteshare:{}".format(pos_voteshare))
        print("#answers with negative score and zero precomputed voteshare:{}".format(zero_voteshare))
        print("#answers with negative score and negative precomputed voteshare:{}".format(neg_voteshare))
        outfile.write("#answers with negative score and positive precomputed voteshare:{}\n".format(pos_voteshare))
        outfile.write("#answers with negative score and zero precomputed voteshare:{}\n".format(zero_voteshare))
        outfile.write("#answers with negative score and negative precomputed voteshare:{}\n".format(neg_voteshare))


def case_2(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    The absolute value of the total vote-scores could be less than the vote-score for particular documents,
    which will result in some vote-shares greater than one. Again, this is possibly harmless.
    """
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    with open('./logs/{}_case2.txt'.format(tag), 'w', encoding="utf8") as outfile:
        num_thread = 0
        for qid in question_list:
            sum_score = 0
            for aid in question_answers.get(qid):
                sum_score += post_scores[aid]
            if any(abs(sum_score) < post_scores[aid] for aid in question_answers.get(qid)):
                print('qid:{}, sum_score:{}, abs(sum_score):{}, answers:{}, answer scores:{}, '
                      'pre_computed_voteshares:{}'.format(
                    qid, sum_score, abs(sum_score),
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                outfile.write('qid:{}, sum_score:{}, abs(sum_score):{}, answers:{}, answer scores:{}, '
                              'pre_computed_voteshares:{}\n'.format(
                    qid, sum_score, abs(sum_score),
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                num_thread += 1
        print("number of questions or threads with at least one answer: {}".format(len(question_list)))
        outfile.write("number of questions or threads with at least one answer: {}\n".format(len(question_list)))
        print("number of threads for case 2: {}".format(num_thread))
        outfile.write("number of threads for case 2: {}\n".format(num_thread))


def case_3(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    In an unusual case, the sum of vote-scores could be negative, resulting in positive vote-share for negative
    vote-scores and negative vote-share for positive vote-scores. Extremely counter-intuitive!  Likely not harmless.
    """
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    with open('./logs/{}_case3.txt'.format(tag), 'w', encoding="utf8") as outfile:
        pos_voteshare, zero_voteshare, neg_voteshare = 0, 0, 0
        pos_score_pos_voteshare, pos_score_zero_voteshare, pos_score_neg_voteshare = 0, 0, 0
        zero_score_pos_voteshare, zero_score_zero_voteshare, zero_score_neg_voteshare = 0, 0, 0
        neg_score_pos_voteshare, neg_score_zero_voteshare, neg_score_neg_voteshare = 0, 0, 0
        num_thread = 0
        for qid in question_list:
            sum_score = 0
            for aid in question_answers.get(qid):
                sum_score += post_scores[aid]
            if sum_score < 0:
                print('qid:{}, sum_score:{}, answers:{}, answer scores:{}, '
                      'answer score/sum-score:{}, pre_computed_voteshares:{}'.format(
                    qid, sum_score,
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [post_scores[aid] / sum_score for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                outfile.write('qid:{}, sum_score:{}, answers:{}, answer scores:{}, '
                              'answer score/sum-score:{}, pre_computed_voteshares:{}\n'.format(
                    qid, sum_score,
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [post_scores[aid] / sum_score for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                num_thread += 1
                for aid in question_answers.get(qid):
                    if aid_voteshares.get(aid, default_voteshare) > 0:
                        pos_voteshare += 1
                    elif aid_voteshares.get(aid, default_voteshare) == 0:
                        zero_voteshare += 1
                    else:
                        neg_voteshare += 1

                    if post_scores[aid] > 0:
                        if aid_voteshares.get(aid, default_voteshare) > 0:
                            pos_score_pos_voteshare += 1
                        elif aid_voteshares.get(aid, default_voteshare) == 0:
                            pos_score_zero_voteshare += 1
                        else:
                            pos_score_neg_voteshare += 1
                    elif post_scores[aid] == 0:
                        if aid_voteshares.get(aid, default_voteshare) > 0:
                            zero_score_pos_voteshare += 1
                        elif aid_voteshares.get(aid, default_voteshare) == 0:
                            zero_score_zero_voteshare += 1
                        else:
                            zero_score_neg_voteshare += 1
                    else:
                        if aid_voteshares.get(aid, default_voteshare) > 0:
                            neg_score_pos_voteshare += 1
                        elif aid_voteshares.get(aid, default_voteshare) == 0:
                            neg_score_zero_voteshare += 1
                        else:
                            neg_score_neg_voteshare += 1
        print("number of questions or threads with at least one answer: {}".format(len(question_list)))
        outfile.write("number of questions or threads with at least one answer: {}\n".format(len(question_list)))
        print("number of threads for case 3: {}".format(num_thread))
        outfile.write("number of threads for case 3: {}\n".format(num_thread))
        print("#answers with positive precomputed voteshare:{}".format(pos_voteshare))
        print("#answers with zero precomputed voteshare:{}".format(zero_voteshare))
        print("#answers with negative precomputed voteshare:{}".format(neg_voteshare))
        outfile.write("#answers with positive precomputed voteshare:{}\n".format(pos_voteshare))
        outfile.write("#answers with zero precomputed voteshare:{}\n".format(zero_voteshare))
        outfile.write("#answers with negative precomputed voteshare:{}\n".format(neg_voteshare))
        outfile.write(
            "#answers with positive score and positive precomputed voteshare:{}\n".format(pos_score_pos_voteshare))
        outfile.write(
            "#answers with positive score and zero precomputed voteshare:{}\n".format(pos_score_zero_voteshare))
        outfile.write(
            "#answers with positive score and negative precomputed voteshare:{}\n".format(pos_score_neg_voteshare))
        outfile.write(
            "#answers with zero score and positive precomputed voteshare:{}\n".format(zero_score_pos_voteshare))
        outfile.write("#answers with zero score and zero precomputed voteshare:{}\n".format(zero_score_zero_voteshare))
        outfile.write(
            "#answers with zero score and negative precomputed voteshare:{}\n".format(zero_score_neg_voteshare))
        outfile.write(
            "#answers with negative score and positive precomputed voteshare:{}\n".format(neg_score_pos_voteshare))
        outfile.write(
            "#answers with negative score and zero precomputed voteshare:{}\n".format(neg_score_zero_voteshare))
        outfile.write(
            "#answers with negative score and negative precomputed voteshare:{}\n".format(neg_score_neg_voteshare))


def case_4(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    The total vote-score could be zero, resulting in useless vote-shares for every document in the thread.
    """
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    with open('./logs/{}_case4.txt'.format(tag), 'w', encoding="utf8") as outfile:
        pos_voteshare, zero_voteshare, neg_voteshare = 0, 0, 0
        num_thread = 0
        for qid in question_list:
            sum_score = 0
            for aid in question_answers.get(qid):
                sum_score += post_scores[aid]
            if sum_score == 0:
                print('qid:{}, sum_score:{}, answers:{}, answer scores:{}, pre_computed_voteshares:{}'.format(
                    qid, sum_score,
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                outfile.write('qid:{}, sum_score:{}, answers:{}, answer scores:{}, pre_computed_voteshares:{}\n'.format(
                    qid, sum_score,
                    [aid for aid in question_answers.get(qid)],
                    [post_scores[aid] for aid in question_answers.get(qid)],
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                num_thread += 1
                for aid in question_answers.get(qid):
                    if aid_voteshares.get(aid, default_voteshare) > 0:
                        pos_voteshare += 1
                    elif aid_voteshares.get(aid, default_voteshare) == 0:
                        zero_voteshare += 1
                    else:
                        neg_voteshare += 1
        print("number of questions or threads with at least one answer: {}".format(len(question_list)))
        outfile.write("number of questions or threads with at least one answer: {}\n".format(len(question_list)))
        print("number of threads for case 4: {}".format(num_thread))
        outfile.write("number of threads for case 4: {}\n".format(num_thread))
        print("#answers with positive precomputed voteshare:{}".format(pos_voteshare))
        print("#answers with zero precomputed voteshare:{}".format(zero_voteshare))
        print("#answers with negative precomputed voteshare:{}".format(neg_voteshare))
        outfile.write("#answers with positive precomputed voteshare:{}\n".format(pos_voteshare))
        outfile.write("#answers with zero precomputed voteshare:{}\n".format(zero_voteshare))
        outfile.write("#answers with negative precomputed voteshare:{}\n".format(neg_voteshare))


def case_5(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    If no document is up-voted, and none down-voted, all vote-scores are zero.
    In this situation, the sum is also zero, and all vote-shares are undefined.
    """
    downvote = load_file('{}_downvotes.txt'.format(tag))  # created by extract_votes.py
    upvote = load_file('{}_upvotes.txt'.format(tag))  # created by extract_votes.py
    upvote_minus_downvote = load_file('{}_upvotes-downvotes.txt'.format(tag))  # created by extract_votes.py
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    with open('./logs/{}_case5.txt'.format(tag), 'w', encoding="utf8") as outfile:
        num_thread = 0
        for qid in question_list:
            forget_this_thread = False
            for aid in question_answers.get(qid):
                if aid not in upvote_minus_downvote or post_scores[aid] != upvote_minus_downvote[aid]:
                    forget_this_thread = True
            if qid not in upvote_minus_downvote or post_scores[qid] != upvote_minus_downvote[qid] or forget_this_thread:
                continue

            answer_upvotes, answer_downvotes = [], []
            for aid in question_answers.get(qid):
                answer_upvotes.append(upvote[aid])
                answer_downvotes.append(downvote[aid])
            # If no answer document is up-voted, and none down-voted, all vote-scores are zero.
            if all(v == 0 for v in answer_upvotes) and all(v == 0 for v in answer_downvotes):
                print('qid:{}, answers:{}, upvotes:{}, downvotes:{}, pre_computed_voteshares:{}'.format(
                    qid, question_answers[qid], answer_upvotes, answer_downvotes,
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                outfile.write('qid:{}, answers:{}, upvotes:{}, downvotes:{}, pre_computed_voteshares:{}\n'.format(
                    qid, question_answers[qid], answer_upvotes, answer_downvotes,
                    [aid_voteshares.get(aid, default_voteshare) for aid in question_answers.get(qid)])
                )
                num_thread += 1
        print("number of questions or threads with at least one answer: {}".format(len(question_list)))
        outfile.write("number of questions or threads with at least one answer: {}\n".format(len(question_list)))
        print("number of threads for case 5: {}".format(num_thread))
        outfile.write("number of threads for case 5: {}\n".format(num_thread))


def num_votes_and_scores(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    If no document is up-voted, and none down-voted, all vote-scores are zero.
    In this situation, the sum is also zero, and all vote-shares are undefined.
    """
    downvote = load_file('{}_downvotes.txt'.format(tag))  # created by extract_votes.py
    upvote = load_file('{}_upvotes.txt'.format(tag))  # created by extract_votes.py
    post_scores = load_file('so_{}_post_score.txt'.format(tag))  # created by 6_get_so_post_score.py

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    sum_upvotes, sum_downvotes = 0, 0
    pos_scores, zero_scores, neg_scores = 0, 0, 0
    for qid in question_list:
        for aid in question_answers.get(qid):
            sum_upvotes += upvote.get(aid, 0)
            sum_downvotes += downvote.get(aid, 0)
            if post_scores[aid] > 0:
                pos_scores += 1
            elif post_scores[aid] == 0:
                zero_scores += 1
            else:
                neg_scores += 1

    print('sum upvotes:{}, sum downvotes:{}'.format(sum_upvotes, sum_downvotes))
    print('#answer with positive score:{}'.format(pos_scores))
    print('#answer with zero score:{}'.format(zero_scores))
    print('#answer with negative score:{}'.format(neg_scores))


def nobari_voteshares(tag, xml_input, pre_computed_voteshares, default_voteshare):
    """
    If no document is up-voted, and none down-voted, all vote-scores are zero.
    In this situation, the sum is also zero, and all vote-shares are undefined.
    """
    aid_voteshares = load_voteshare_file(pre_computed_voteshares)

    question_list, answer_list, answer_parent, answer_score, question_answers = load_post_xml(xml_input)

    pos_voteshare, zero_voteshare, neg_voteshare = 0, 0, 0
    above_one_voteshare, lower_minus_one_voteshare = 0, 0
    one_voteshare, minus_one_voteshare = 0, 0
    for qid in question_list:
        for aid in question_answers.get(qid):
            if aid_voteshares.get(aid, default_voteshare) > 0:
                pos_voteshare += 1
                if aid_voteshares.get(aid, default_voteshare) > 1:
                    above_one_voteshare += 1
                if aid_voteshares.get(aid, default_voteshare) == 1:
                    one_voteshare += 1
            elif aid_voteshares.get(aid, default_voteshare) == 0:
                zero_voteshare += 1
            else:
                neg_voteshare += 1
                if aid_voteshares.get(aid, default_voteshare) == -1:
                    minus_one_voteshare += 1
                if aid_voteshares.get(aid, default_voteshare) < -1:
                    lower_minus_one_voteshare += 1

    print("#answers with positive precomputed voteshare:{}".format(pos_voteshare))
    print("#answers with zero precomputed voteshare:{}".format(zero_voteshare))
    print("#answers with negative precomputed voteshare:{}".format(neg_voteshare))
    print("#answers with precomputed voteshare above 1:{}".format(above_one_voteshare))
    print("#answers with precomputed voteshare equal 1:{}".format(one_voteshare))
    print("#answers with precomputed voteshare equal -1:{}".format(minus_one_voteshare))
    print("#answers with precomputed voteshare lower than -1:{}".format(lower_minus_one_voteshare))


if __name__ == '__main__':
    print('Hi!')

    # case_1(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_1(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_2(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_2(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_3(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_3(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_4(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_4(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_5(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # case_5(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # num_votes_and_scores(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # num_votes_and_scores(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # nobari_voteshares(
    #     tag='java',
    #     xml_input='JavaPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/java/java_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
    # nobari_voteshares(
    #     tag='php',
    #     xml_input='PhpPosts.xml',
    #     pre_computed_voteshares='../../evaluate/VoteShare/php/php_vote_share_nobari.csv',
    #     default_voteshare=0
    # )
