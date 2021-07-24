import os
import re
import pickle
import statistics

JAVA_TOP_TAGS = [
    "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp", "string",
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
    "mysql", "javascript", "html", "jquery", "arrays", "ajax", "wordpress", "sql", "codeigniter", "regex", "forms",
    "json", "apache", "database", ".htaccess", "symfony2", "laravel", "xml", "zend-framework", "curl", "session", "pdo",
    "css", "mysqli", "facebook", "cakephp", "email", "magento", "yii", "laravel-4", "oop", "string", "post", "image",
    "function", "variables", "api", "date", "mod-rewrite", "android", "security", "foreach", "multidimensional-array",
    "redirect", "url", "class", "validation", "java", "doctrine2", "linux", "file-upload", "joomla", "cookies", "loops",
    "facebook-graph-api", "file", "drupal", "soap", "datetime", "login", "preg-replace", "parsing", "csv",
    "if-statement", "zend-framework2", "html5", "upload", "paypal", "preg-match", "sorting", "phpmyadmin", "search",
    "get", "sql-server", "doctrine", "performance", "web-services", "table", "pdf", "utf-8", "simplexml", "object",
    "phpunit", "mongodb", "dom", "select", "http", "include", "authentication", "caching", "cron", "pagination",
    "twitter", "xampp", "python", "rest", "encryption", "wordpress-plugin", "gd", "smarty"
]
for target_tag, so_input_path in [("java", "./data/JavaPosts.xml"), ("php", "./data/PhpPosts.xml")]:
    if target_tag == "java":
        TOP_TAGS = JAVA_TOP_TAGS
    elif target_tag == "php":
        TOP_TAGS = PHP_TOP_TAGS
    else:
        exit(1)

    so_output_path = "./data/"
    if not os.path.exists(so_output_path):
        os.makedirs(so_output_path)

    print("Start reading stack overflow data...")
    post_tags = {}
    question_tags = {}
    post_tag_freq, question_tag_freq = {}, {}
    id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    tags_regex = re.compile("(?<=Tags=\")(?P<Tags>.*?)(?=\" )")
    post_type_id_regex = re.compile("(?<=PostTypeId=\")(?P<PostTypeId>.*?)(?=\" )")
    with open(so_input_path, encoding='utf8') as posts_file:
        for line in posts_file:
            post_id = id_regex.search(line).group('Id')
            post_type_id = post_type_id_regex.search(line).group('PostTypeId')
            if not tags_regex.search(line):
                continue
            tags = tags_regex.search(line).group('Tags').replace("&lt;", "<").replace("&gt;", ">")[1:-1].split("><")
            tags = [tag for tag in tags if tag in TOP_TAGS]
            if tags:
                print(post_id, tags)
                post_tags[post_id] = tags
                for tag in tags:
                    if tag in post_tag_freq:
                        post_tag_freq[tag] += 1
                    else:
                        post_tag_freq[tag] = 1

                if post_type_id == "1":
                    question_tags[post_id] = tags
                    for tag in tags:
                        if tag in question_tag_freq:
                            question_tag_freq[tag] += 1
                        else:
                            question_tag_freq[tag] = 1

    print("End!")

    with open(os.path.join(so_output_path, "so_{}_post_tags.pkl".format(target_tag)), 'wb') as output:
        pickle.dump(post_tags, output)
        print("Number of posts: {}".format(len(post_tags)))
    with open(os.path.join(so_output_path, "so_{}_question_tags.pkl".format(target_tag)), 'wb') as output:
        pickle.dump(question_tags, output)
        print("Number of question: {}".format(len(question_tags)))
    with open(os.path.join(so_output_path, "so_{}_answer_tags.pkl".format(target_tag)), 'wb') as output:
        answer_tags = {pid: post_tags[pid] for pid in post_tags if pid not in question_tags}
        pickle.dump(answer_tags, output)
        print("Number of answers: {}".format(len(answer_tags)))

    print("\nPosts:")
    print("Number of posts: {}".format(len(post_tags)))
    tags_count = [len(post_tags[pid]) for pid in post_tags]
    length_count = {l: tags_count.count(l) for l in set(tags_count)}
    print("Average number of tags=", statistics.mean(tags_count))
    print("Median number of tags=", statistics.median(tags_count))
    print("Minimum number of tags={}, number of posts={}".format(min(tags_count), length_count[min(tags_count)]))
    print("Maximum number of tags={}, number of posts={}".format(max(tags_count), length_count[max(tags_count)]))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum frequency={}, with length={}".format(max_count, length_max_count))
    print("Tag frequency in posts:")
    for tag in TOP_TAGS:
        print(tag, post_tag_freq[tag])

    print("\nQuestions:")
    print("Number of question: {}".format(len(question_tags)))
    tags_count = [len(question_tags[pid]) for pid in question_tags]
    length_count = {l: tags_count.count(l) for l in set(tags_count)}
    print("Average number of tags=", statistics.mean(tags_count))
    print("Median number of tags=", statistics.median(tags_count))
    print("Minimum number of tags={}, number of posts={}".format(min(tags_count), length_count[min(tags_count)]))
    print("Maximum number of tags={}, number of posts={}".format(max(tags_count), length_count[max(tags_count)]))
    max_count = max(length_count.values())
    length_max_count = list(length_count.keys())[list(length_count.values()).index(max_count)]
    print("Maximum frequency={}, with length={}".format(max_count, length_max_count))
    print("Tag frequency in questions:")
    for tag in TOP_TAGS:
        print(tag, question_tag_freq[tag])

'''
tag: java
Number of posts: 1605653
Number of question: 570815
Number of answers: 1034838

Posts:
Number of posts: 1605653
Average number of tags= 1.4595538388431373
Median number of tags= 1
Minimum number of tags=1, number of posts=1013288
Maximum number of tags=4, number of posts=15297
Maximum frequency=1013288, with length=1
Tag frequency in posts:
android 272112
swing 129993
eclipse 93366
spring 83096
hibernate 60001
arrays 66813
multithreading 58038
xml 45029
jsp 43279
string 59235
servlets 37574
maven 32389
java-ee 33828
mysql 32183
spring-mvc 29878
json 30578
regex 39735
tomcat 28771
jpa 27976
jdbc 29041
javascript 25119
arraylist 33922
web-services 21670
sql 23882
generics 28764
netbeans 20415
sockets 19903
user-interface 21366
jar 19998
html 17916
jsf 16313
database 19034
file 20228
google-app-engine 15351
gwt 16272
junit 18086
exception 20426
algorithm 21696
rest 14153
class 20421
performance 21068
applet 13736
image 13654
jtable 13242
c# 19160
jframe 13592
collections 20464
c++ 17390
methods 17424
oop 19280
linux 12282
nullpointerexception 13553
jaxb 10757
parsing 13573
oracle 11635
concurrency 15158
php 12158
jpanel 11293
jboss 9818
object 14998
ant 11292
date 14907
selenium 10528
javafx 8868
jvm 13553
list 14598
struts2 9944
hashmap 14580
sorting 13718
awt 10118
http 9985
inheritance 14442
reflection 12283
hadoop 8578
windows 10330
loops 13774
unit-testing 11584
sqlite 9322
design-patterns 12914
serialization 10300
security 9395
intellij-idea 8748
file-io 10913
logging 9296
swt 7562
apache 7306
annotations 8881
jquery 7614
jersey 7326
scala 8716
libgdx 6499
osx 7892
encryption 7702
spring-security 6890
log4j 8200
python 9442
jni 7309
soap 6162
interface 11153
io 8828

Questions:
Number of question: 570815
Average number of tags= 1.4731813284514248
Median number of tags= 1
Minimum number of tags=1, number of posts=355381
Maximum number of tags=4, number of posts=5896
Maximum frequency=355381, with length=1
Tag frequency in questions:
android 106861
swing 48723
eclipse 34584
spring 33534
hibernate 24008
arrays 18926
multithreading 18217
xml 17480
jsp 16449
string 15091
servlets 14029
maven 13469
java-ee 12435
mysql 12336
spring-mvc 12186
json 12014
regex 11523
tomcat 11434
jpa 11236
jdbc 10319
javascript 9958
arraylist 9774
web-services 9433
sql 8720
generics 8317
netbeans 8198
sockets 7774
user-interface 7728
jar 7247
html 6948
jsf 6835
database 6823
file 6598
google-app-engine 6442
gwt 6362
junit 6261
exception 6189
algorithm 6106
rest 5997
class 5990
performance 5795
applet 5767
image 5403
jtable 5214
c# 5197
jframe 5161
collections 5107
c++ 4911
methods 4882
oop 4698
linux 4571
nullpointerexception 4555
jaxb 4534
parsing 4466
oracle 4448
concurrency 4412
php 4407
jpanel 4385
jboss 4302
object 4254
ant 4248
date 4238
selenium 4198
javafx 4155
jvm 4142
list 4068
struts2 4060
hashmap 4059
sorting 3966
awt 3901
http 3879
inheritance 3870
reflection 3848
hadoop 3843
windows 3799
loops 3789
unit-testing 3756
sqlite 3694
design-patterns 3652
serialization 3595
security 3576
intellij-idea 3558
file-io 3401
logging 3345
swt 3331
apache 3314
annotations 3270
jquery 3175
jersey 3175
scala 3120
libgdx 3082
osx 3034
encryption 3012
spring-security 3004
log4j 3002
python 2968
jni 2958
soap 2940
interface 2933
io 2933
'''

'''
tag: php
Posts:
Number of posts: 714476
Average number of tags= 3.0931255913424662
Median number of tags= 3.0
Minimum number of tags=1, number of posts=51564
Maximum number of tags=5, number of posts=97682
Maximum frequency=224216, with length=3
Tag frequency in posts:
php 714476
mysql 126008
javascript 57867
html 51529
jquery 49484
arrays 32580
ajax 29749
wordpress 27931
sql 27749
codeigniter 24056
regex 18426
forms 17015
json 16757
apache 14249
database 13714
.htaccess 12382
symfony2 11935
laravel 11626
xml 11407
zend-framework 10921
curl 10870
session 10856
pdo 10612
css 9779
mysqli 9635
facebook 9424
cakephp 9424
email 9142
magento 8886
yii 7851
laravel-4 6984
oop 6968
string 6807
post 6804
image 6568
function 5882
variables 5727
api 5717
date 5546
mod-rewrite 5523
android 5475
security 5296
foreach 5088
multidimensional-array 5042
redirect 4631
url 4571
class 4565
validation 4478
java 4407
doctrine2 4369
linux 4363
file-upload 4361
joomla 4235
cookies 4233
loops 4142
facebook-graph-api 4089
file 3970
drupal 3542
soap 3480
datetime 3459
login 3363
preg-replace 3319
parsing 3272
csv 3263
if-statement 3253
zend-framework2 3236
html5 3206
upload 3165
paypal 3145
preg-match 3055
sorting 3035
phpmyadmin 3022
search 2952
get 2867
sql-server 2854
doctrine 2819
performance 2807
web-services 2801
table 2797
pdf 2688
utf-8 2646
simplexml 2644
object 2639
phpunit 2632
mongodb 2598
dom 2581
select 2559
http 2554
include 2504
authentication 2472
caching 2425
cron 2372
pagination 2325
twitter 2310
xampp 2310
python 2292
rest 2290
encryption 2164
wordpress-plugin 2143
gd 2055
smarty 2042

Questions:
Number of question: 714476
Average number of tags= 3.0931255913424662
Median number of tags= 3.0
Minimum number of tags=1, number of posts=51564
Maximum number of tags=5, number of posts=97682
Maximum frequency=224216, with length=3
Tag frequency in questions:
php 714476
mysql 126008
javascript 57867
html 51529
jquery 49484
arrays 32580
ajax 29749
wordpress 27931
sql 27749
codeigniter 24056
regex 18426
forms 17015
json 16757
apache 14249
database 13714
.htaccess 12382
symfony2 11935
laravel 11626
xml 11407
zend-framework 10921
curl 10870
session 10856
pdo 10612
css 9779
mysqli 9635
facebook 9424
cakephp 9424
email 9142
magento 8886
yii 7851
laravel-4 6984
oop 6968
string 6807
post 6804
image 6568
function 5882
variables 5727
api 5717
date 5546
mod-rewrite 5523
android 5475
security 5296
foreach 5088
multidimensional-array 5042
redirect 4631
url 4571
class 4565
validation 4478
java 4407
doctrine2 4369
linux 4363
file-upload 4361
joomla 4235
cookies 4233
loops 4142
facebook-graph-api 4089
file 3970
drupal 3542
soap 3480
datetime 3459
login 3363
preg-replace 3319
parsing 3272
csv 3263
if-statement 3253
zend-framework2 3236
html5 3206
upload 3165
paypal 3145
preg-match 3055
sorting 3035
phpmyadmin 3022
search 2952
get 2867
sql-server 2854
doctrine 2819
performance 2807
web-services 2801
table 2797
pdf 2688
utf-8 2646
simplexml 2644
object 2639
phpunit 2632
mongodb 2598
dom 2581
select 2559
http 2554
include 2504
authentication 2472
caching 2425
cron 2372
pagination 2325
twitter 2310
xampp 2310
python 2292
rest 2290
encryption 2164
wordpress-plugin 2143
gd 2055
smarty 2042
'''
