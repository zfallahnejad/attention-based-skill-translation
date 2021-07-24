import os
import re

so_input_path = "./data/Posts.xml"
so_output_path = "./data/"
if not os.path.exists(so_output_path):
    os.makedirs(so_output_path)

for target_tag, outfile_name in [("java", "JavaPosts.xml"), ("php", "PhpPosts.xml")]:
    print(target_tag, outfile_name)
    questions = set()
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    parent_id_regex = re.compile("(?<=ParentId=\")(?P<ParentId>.*?)(?=\" )")
    tags_regex = re.compile("(?<=Tags=\")(?P<Tags>.*?)(?=\" )")
    print("Loading posts in order to find question ids...")
    with open(so_input_path, encoding='utf8') as posts_file:
        for line in posts_file:
            if "<row" not in line:
                continue
            post_id = id_regex.search(line).group('Id')
            if post_type_id_regex.search(line):
                post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
            else:
                post_type_id = -1

            if tags_regex.search(line):
                tags = tags_regex.search(line).group('Tags').replace("&lt;", "<").replace("&gt;", ">")[1:-1].split("><")
            else:
                tags = []

            if post_type_id == 1 and target_tag in tags:
                questions.add(post_id)
    print("number of {} questions: {}".format(target_tag, len(questions)))

    answers = 0
    with open(so_output_path + outfile_name, "w", encoding='utf8') as out_file:
        with open(so_input_path, encoding='utf8') as posts_file:
            for line in posts_file:
                if "<row" not in line:
                    continue
                post_id = id_regex.search(line).group('Id')
                if post_type_id_regex.search(line):
                    post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
                else:
                    post_type_id = -1

                if parent_id_regex.search(line):
                    parent_id = parent_id_regex.search(line).group('ParentId')
                else:
                    parent_id = -1

                if post_type_id == 1 and post_id in questions:
                    out_file.write(line)
                elif post_type_id == 2 and parent_id in questions:
                    out_file.write(line)
                    answers += 1
    print("number of {} answers: {}".format(target_tag, answers))
