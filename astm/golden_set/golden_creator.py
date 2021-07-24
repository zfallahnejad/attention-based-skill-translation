import os
import re
import pickle

JAVA_TOP_TAGS = [
    "java", "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp", "string",
    "servlets", "maven", "java-ee", "mysql", "spring-mvc", "json", "regex", "tomcat", "jpa", "jdbc", "javascript",
    "arraylist", "web-services", "sql", "generics", "netbeans", "sockets", "user-interface", "jar", "html", "jsf",
    "database", "file", "google-app-engine", "gwt", "junit", "exception", "algorithm", "rest", "class", "performance",
    "applet", "image", "jtable", "c#", "jframe", "collections", "c++", "methods", "oop", "linux",
    "nullpointerexception", "jaxb", "parsing", "oracle", "concurrency", "php", "jpanel", "jboss", "object", "ant",
    "date", "selenium", "javafx", "jvm", "list", "struts2", "hashmap", "sorting", "awt", "http", "inheritance",
    "reflection", "hadoop", "windows", "loops", "unit-testing", "sqlite", "design-patterns", "serialization",
    "security", "intellij-idea", "file-io", "logging", "swt", "apache", "annotations", "jquery", "jersey", "scala",
    "libgdx", "osx", "encryption", "spring-security", "log4j", "python", "jni", "soap", "interface", "io"
]
PHP_TOP_TAGS = [
    "php", "mysql", "javascript", "html", "jquery", "arrays", "ajax", "wordpress", "sql", "codeigniter", "regex",
    "forms", "json", "apache", "database", ".htaccess", "symfony2", "laravel", "xml", "zend-framework", "curl",
    "session", "pdo", "css", "mysqli", "facebook", "cakephp", "email", "magento", "yii", "laravel-4", "oop", "string",
    "post", "image", "function", "variables", "api", "date", "mod-rewrite", "android", "security", "foreach",
    "multidimensional-array", "redirect", "url", "class", "validation", "java", "doctrine2", "linux", "file-upload",
    "joomla", "cookies", "loops", "facebook-graph-api", "file", "drupal", "soap", "datetime", "login", "preg-replace",
    "parsing", "csv", "if-statement", "zend-framework2", "html5", "upload", "paypal", "preg-match", "sorting",
    "phpmyadmin", "search", "get", "sql-server", "doctrine", "performance", "web-services", "table", "pdf", "utf-8",
    "simplexml", "object", "phpunit", "mongodb", "dom", "select", "http", "include", "authentication", "caching",
    "cron", "pagination", "twitter", "xampp", "python", "rest", "encryption", "wordpress-plugin", "gd", "smarty"
]

for target_tag, so_input_path, accepted_answer_thresh, acceptance_thresh in [("java", "../data/JavaPosts.xml", 10, 0.4),
                                                                             ("php", "../data/PhpPosts.xml", 7, 0.4)]:
    if target_tag == "java":
        TOP_TAGS = JAVA_TOP_TAGS
    elif target_tag == "php":
        TOP_TAGS = PHP_TOP_TAGS

    answer_owner = {}
    answer_question = {}
    question_tags = {}
    accepted_answers = set()

    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    owner_regex = re.compile("(?<=OwnerUserId=\")(?P<OwnerUserId>.*?)(?=\" )")
    parent_id_regex = re.compile("(?<=ParentId=\")(?P<ParentId>.*?)(?=\" )")
    accepted_answer_id_regex = re.compile("(?<=AcceptedAnswerId=\")(?P<AcceptedAnswerId>.*?)(?=\" )")
    tags_regex = re.compile("(?<=Tags=\")(?P<Tags>.*?)(?=\" )")
    with open(so_input_path, encoding='utf8') as posts_file:
        for line in posts_file:
            if "<row" not in line:
                continue
            post_id = id_regex.search(line).group('Id')
            if post_type_id_regex.search(line):
                post_type_id = int(post_type_id_regex.search(line).group('PostTypeId'))
            else:
                continue
            if post_type_id == 1:
                # question
                if tags_regex.search(line):
                    tags = tags_regex.search(line).group('Tags').replace("&lt;", "<").replace("&gt;", ">")[1:-1].split(
                        "><")
                else:
                    tags = []
                question_tags[post_id] = tags

                if accepted_answer_id_regex.search(line):
                    accepted_answer_id = accepted_answer_id_regex.search(line).group('AcceptedAnswerId')
                    accepted_answers.add(accepted_answer_id)
            elif post_type_id == 2:
                # answer
                if owner_regex.search(line):
                    owner_id = owner_regex.search(line).group('OwnerUserId')
                    answer_owner[post_id] = owner_id

                if parent_id_regex.search(line):
                    parent_id = parent_id_regex.search(line).group('ParentId')
                    answer_question[post_id] = parent_id

    print(len(answer_owner))  # java:1503487 php:1289405
    print(len(answer_question))  # java:1510812 php:1298107
    print(len(question_tags))  # java:810071 php:714476
    print(len(accepted_answers))  # java:452233 php:408077

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    with open(os.path.join("tmp", target_tag + '_data.pkl'), 'wb') as outfile:
        pickle.dump((answer_owner, answer_question, question_tags, accepted_answers), outfile)

    ################################################################################################################

    with open(os.path.join("tmp", target_tag + '_data.pkl'), "rb") as infile:
        answer_owner, answer_question, question_tags, accepted_answers = pickle.load(infile)

    tag_user_accepted_answer = {t: {} for t in TOP_TAGS}
    tag_user_not_accepted_answer = {t: {} for t in TOP_TAGS}
    for answer, user in answer_owner.items():
        related_tags = [t for t in question_tags[answer_question[answer]] if t in TOP_TAGS]
        if answer in accepted_answers:
            for tag in related_tags:
                if user in tag_user_accepted_answer[tag]:
                    tag_user_accepted_answer[tag][user] += 1
                else:
                    tag_user_accepted_answer[tag][user] = 1
        else:
            for tag in related_tags:
                if user in tag_user_not_accepted_answer[tag]:
                    tag_user_not_accepted_answer[tag][user] += 1
                else:
                    tag_user_not_accepted_answer[tag][user] = 1

    if not os.path.exists(os.path.join("golden", target_tag)):
        os.makedirs(os.path.join("golden", target_tag))
    if not os.path.exists(os.path.join("golden", target_tag + "-all")):
        os.makedirs(os.path.join("golden", target_tag + "-all"))
    for tag in TOP_TAGS:
        sorted_user_num_accepted = sorted(tag_user_accepted_answer[tag].items(), key=lambda x: x[1], reverse=True)
        outfile1 = open(os.path.join("golden", target_tag + "-all", tag + ".csv"), "w", encoding="utf8")
        outfile2 = open(os.path.join("golden", target_tag, tag + ".csv"), "w", encoding="utf8")
        for user, score in sorted_user_num_accepted:
            if user in tag_user_not_accepted_answer[tag]:
                acceptance_ratio = score / (score + tag_user_not_accepted_answer[tag][user])
                outfile1.write(
                    "{},{},{},{}\n".format(user, score, tag_user_not_accepted_answer[tag][user], acceptance_ratio))
                if score >= accepted_answer_thresh and acceptance_ratio > acceptance_thresh:
                    outfile2.write(
                        "{},{},{},{}\n".format(user, score, tag_user_not_accepted_answer[tag][user], acceptance_ratio))
            else:
                acceptance_ratio = 1.0
                outfile1.write("{},{},{},{}\n".format(user, score, 0, acceptance_ratio))
                if score >= accepted_answer_thresh:
                    outfile2.write("{},{},{},{}\n".format(user, score, 0, acceptance_ratio))
