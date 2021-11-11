import re

post_id_main_tag = {}
post_ids = set()
for target_tag, infile_name in [("java", "JavaPosts.xml"), ("php", "PhpPosts.xml")]:
    print("target_tag:{}".format(target_tag))
    Id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    with open(infile_name, encoding='utf8') as posts_file:
        for line in posts_file:
            post_id = Id_regex.search(line).group('Id')
            print('post_id: {}'.format(post_id))
            post_ids.add(post_id)
            post_id_main_tag[post_id] = target_tag

vote_filename = "Votes.xml"
print("Loading votes")
post_upvotes, post_downvotes = {pid: 0 for pid in post_ids}, {pid: 0 for pid in post_ids}
PostId_regex = re.compile("(?<=PostId=\")(?P<PostId>.*?)(?=\" )")
VoteTypeId_regex = re.compile("(?<=VoteTypeId=\")(?P<VoteTypeId>.*?)(?=\" )")
with open(vote_filename, encoding='utf8') as vfile:
    for i, line in enumerate(vfile):
        if not line.strip().startswith('<row'):
            continue

        post_id = PostId_regex.search(line).group('PostId')
        if post_id not in post_ids:
            continue
        print('i: {}, post_id: {}'.format(i, post_id))

        # vote_type_id = int(VoteTypeId_regex.search(line).group('VoteTypeId'))
        if 'VoteTypeId="2"' in line:  # vote_type_id == 2:
            post_upvotes[post_id] += 1
        if 'VoteTypeId="3"' in line:  # vote_type_id == 3:
            post_downvotes[post_id] += 1

for target_tag in ["java", "php"]:
    print("target_tag:{}".format(target_tag))
    with open("{}_upvotes.txt".format(target_tag), "w", encoding='utf8') as out_file:
        for pid in post_ids:
            if post_id_main_tag[pid] == target_tag:
                out_file.write('{},{}\n'.format(pid, post_upvotes[pid]))
    with open("{}_downvotes.txt".format(target_tag), "w", encoding='utf8') as out_file:
        for pid in post_ids:
            if post_id_main_tag[pid] == target_tag:
                out_file.write('{},{}\n'.format(pid, post_downvotes[pid]))
    with open("{}_upvotes-downvotes.txt".format(target_tag), "w", encoding='utf8') as out_file:
        for pid in post_ids:
            if post_id_main_tag[pid] == target_tag:
                out_file.write('{},{}\n'.format(pid, post_upvotes[pid] - post_downvotes[pid]))
    print("End {}!".format(target_tag))
