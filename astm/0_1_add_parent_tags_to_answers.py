import os
import re

so_input_path = "./data/"
so_output_path = "./data/"
if not os.path.exists(so_output_path):
    os.makedirs(so_output_path)

for target_tag, outfile_name, temp_file_name in [("java", "JavaPosts.xml", "JavaPosts_2.xml"),
                                                 ("php", "PhpPosts.xml", "PhpPosts_2.xml")]:
    os.rename(os.path.join(so_input_path, outfile_name), os.path.join(so_input_path, temp_file_name))

    question_tags = {}
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    tags_regex = re.compile("(?<=Tags=\")(?P<Tags>.*?)(?=\" )")
    parent_id_regex = re.compile("(?<=ParentId=\")(?P<ParentId>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    with open(os.path.join(so_input_path, temp_file_name), encoding='utf8') as posts_file:
        for line in posts_file:
            if "<row" not in line:
                continue
            post_id = id_regex.search(line).group('Id')
            if post_type_id_regex.search(line):
                post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
            else:
                continue
            if post_type_id == 1:
                question_tags[post_id] = tags_regex.search(line).group('Tags')

    with open(os.path.join(so_input_path, outfile_name), "w", encoding='utf8') as out_file:
        with open(os.path.join(so_input_path, temp_file_name), encoding='utf8') as posts_file:
            for line in posts_file:
                if "<row" not in line:
                    continue
                line = line.strip()
                post_id = id_regex.search(line).group('Id')
                if post_type_id_regex.search(line):
                    post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
                    if post_type_id == 2:
                        if parent_id_regex.search(line):
                            parent_id = parent_id_regex.search(line).group('ParentId')
                            if parent_id in question_tags:
                                line = re.sub(r'/>$', "Tags=\"{}\" />\n".format(question_tags[parent_id]), line).strip()
                                out_file.write(line + "\n")
                            else:
                                out_file.write(line + "\n")
                        else:
                            out_file.write(line + "\n")
                    else:
                        out_file.write(line + "\n")
                else:
                    out_file.write(line + "\n")
