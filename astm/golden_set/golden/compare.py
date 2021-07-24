import os

# target_tag, so_input_path = "java", "../data/JavaPosts.xml"
target_tag, so_input_path = "php", "../data/PhpPosts.xml"
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
if target_tag == "java":
    TOP_TAGS = JAVA_TOP_TAGS
elif target_tag == "php":
    TOP_TAGS = PHP_TOP_TAGS

# neshati
neshati_tag_experts = {t: set() for t in TOP_TAGS}
for tag in TOP_TAGS:
    with open(os.path.join("../neshati", target_tag, tag + ".csv")) as infile:
        for line in infile:
            line = line.strip().split(",")
            neshati_tag_experts[tag].add(line[0])
    # print(tag, len(neshati_tag_experts[tag]))

# me
me_tag_experts = {t: set() for t in TOP_TAGS}
for tag in TOP_TAGS:
    with open(os.path.join(target_tag, tag + ".csv")) as infile:
        for line in infile:
            line = line.strip().split(",")
            me_tag_experts[tag].add(line[0])
    # print(tag, len(me_tag_experts[tag]))

for tag in TOP_TAGS:
    print(tag, len(neshati_tag_experts[tag]), len(me_tag_experts[tag]),
          neshati_tag_experts[tag].difference(me_tag_experts[tag]),
          me_tag_experts[tag].difference(neshati_tag_experts[tag]),
          )