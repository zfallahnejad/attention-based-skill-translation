package Utility;

public class Constants {

    public static final String JavaXMLInput = "../astm/data/JavaPosts.xml";
    //public static final String JavaIndexDirectory = "./Indexes/JavaIndex_StandardAnalyzer";
    public static final String JavaIndexDirectory2 = "./Indexes/JavaIndex2_StandardAnalyzer";

    public static final String PhpXMLInput = "../astm/data/PhpPosts.xml";
    //public static final String PhpIndexDirectory = "./Indexes/PhpIndex_StandardAnalyzer";
    public static final String PhpIndexDirectory2 = "./Indexes/PhpIndex2_StandardAnalyzer";

    public static final String[] Java_TopTags = new String[]{
            "java", "android", "swing", "spring", "eclipse", "hibernate", "multithreading", "arrays", "xml", "jsp",
            "maven", "servlets", "string", "mysql", "spring-mvc", "java-ee", "json", "jpa", "tomcat", "regex", "jdbc",
            "web-services", "arraylist", "sql", "javascript", "sockets", "generics", "netbeans", "user-interface",
            "jar", "file", "junit", "database", "google-app-engine", "exception", "html", "rest", "algorithm", "jsf",
            "gwt", "class", "performance", "image", "applet", "jframe", "jtable", "nullpointerexception", "methods",
            "linux", "collections", "jpanel"
    };

    // Top 101 tags of java collection questions
    // Order of this list is based on tag frequency (number of question with these tags)
    public static final String[] Java_TopTags2 = new String[]{
            "java", "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp",
            "string", "servlets", "maven", "java-ee", "mysql", "spring-mvc", "json", "regex", "tomcat", "jpa", "jdbc",
            "javascript", "arraylist", "web-services", "sql", "generics", "netbeans", "sockets", "user-interface",
            "jar", "html", "jsf", "database", "file", "google-app-engine", "gwt", "junit", "exception", "algorithm",
            "rest", "class", "performance", "applet", "image", "jtable", "c#", "jframe", "collections", "c++",
            "methods", "oop", "linux", "nullpointerexception", "jaxb", "parsing", "oracle", "concurrency", "php",
            "jpanel", "jboss", "object", "ant", "date", "selenium", "javafx", "jvm", "list", "struts2", "hashmap",
            "sorting", "awt", "http", "inheritance", "reflection", "hadoop", "windows", "loops", "unit-testing",
            "sqlite", "design-patterns", "serialization", "security", "intellij-idea", "file-io", "logging", "swt",
            "apache", "annotations", "jquery", "jersey", "scala", "libgdx", "osx", "encryption", "spring-security",
            "log4j", "python", "jni", "soap", "interface", "io"
    };
    // same values as the Java_TopTags2 except java tag
    public static final String[] Java_TopTags3 = new String[]{
            "android", "swing", "eclipse", "spring", "hibernate", "arrays", "multithreading", "xml", "jsp",
            "string", "servlets", "maven", "java-ee", "mysql", "spring-mvc", "json", "regex", "tomcat", "jpa", "jdbc",
            "javascript", "arraylist", "web-services", "sql", "generics", "netbeans", "sockets", "user-interface",
            "jar", "html", "jsf", "database", "file", "google-app-engine", "gwt", "junit", "exception", "algorithm",
            "rest", "class", "performance", "applet", "image", "jtable", "c#", "jframe", "collections", "c++",
            "methods", "oop", "linux", "nullpointerexception", "jaxb", "parsing", "oracle", "concurrency", "php",
            "jpanel", "jboss", "object", "ant", "date", "selenium", "javafx", "jvm", "list", "struts2", "hashmap",
            "sorting", "awt", "http", "inheritance", "reflection", "hadoop", "windows", "loops", "unit-testing",
            "sqlite", "design-patterns", "serialization", "security", "intellij-idea", "file-io", "logging", "swt",
            "apache", "annotations", "jquery", "jersey", "scala", "libgdx", "osx", "encryption", "spring-security",
            "log4j", "python", "jni", "soap", "interface", "io"
    };

    // Top 101 tags of php collection questions
    // Order of this list is based on tag frequency (number of question with these tags)
    public static final String[] Php_TopTags2 = new String[]{
            "php", "mysql", "javascript", "html", "jquery", "arrays", "ajax", "wordpress", "sql", "codeigniter",
            "regex", "forms", "json", "apache", "database", ".htaccess", "symfony2", "laravel", "xml", "zend-framework",
            "curl", "session", "pdo", "css", "mysqli", "facebook", "cakephp", "email", "magento", "yii", "laravel-4",
            "oop", "string", "post", "image", "function", "variables", "api", "date", "mod-rewrite", "android",
            "security", "foreach", "multidimensional-array", "redirect", "url", "class", "validation", "java",
            "doctrine2", "linux", "file-upload", "joomla", "cookies", "loops", "facebook-graph-api", "file", "drupal",
            "soap", "datetime", "login", "preg-replace", "parsing", "csv", "if-statement", "zend-framework2", "html5",
            "upload", "paypal", "preg-match", "sorting", "phpmyadmin", "search", "get", "sql-server", "doctrine",
            "performance", "web-services", "table", "pdf", "utf-8", "simplexml", "object", "phpunit", "mongodb", "dom",
            "select", "http", "include", "authentication", "caching", "cron", "pagination", "twitter", "xampp",
            "python", "rest", "encryption", "wordpress-plugin", "gd", "smarty"
    };
    public static final String[] Php_TopTags3 = new String[]{
            "mysql", "javascript", "html", "jquery", "arrays", "ajax", "wordpress", "sql", "codeigniter",
            "regex", "forms", "json", "apache", "database", ".htaccess", "symfony2", "laravel", "xml", "zend-framework",
            "curl", "session", "pdo", "css", "mysqli", "facebook", "cakephp", "email", "magento", "yii", "laravel-4",
            "oop", "string", "post", "image", "function", "variables", "api", "date", "mod-rewrite", "android",
            "security", "foreach", "multidimensional-array", "redirect", "url", "class", "validation", "java",
            "doctrine2", "linux", "file-upload", "joomla", "cookies", "loops", "facebook-graph-api", "file", "drupal",
            "soap", "datetime", "login", "preg-replace", "parsing", "csv", "if-statement", "zend-framework2", "html5",
            "upload", "paypal", "preg-match", "sorting", "phpmyadmin", "search", "get", "sql-server", "doctrine",
            "performance", "web-services", "table", "pdf", "utf-8", "simplexml", "object", "phpunit", "mongodb", "dom",
            "select", "http", "include", "authentication", "caching", "cron", "pagination", "twitter", "xampp",
            "python", "rest", "encryption", "wordpress-plugin", "gd", "smarty"
    };

    public static final String JavaGoldenSetDirectory = "./GoldenSet/java/";
    public static final String PhpGoldenSetDirectory = "./GoldenSet/php/";
    public static final String JavaGoldenSetDirectory_Python = "../astm/golden_set/golden/java/";
    public static final String PhpGoldenSetDirectory_Python = "../astm/golden_set/golden/php/";
    // use the following directory
    public static final String JavaGoldenSetDirectory_Neshati = "../astm/golden_set/neshati/java/";
    public static final String PhpGoldenSetDirectory_Neshati = "../astm/golden_set/neshati/php/";

    public static final String Results_Directory = "./Results/";
    public static final String EvaluationResultsDirectory = "./EvaluationResults/";
    public static final String Results2_Directory = "./Results2/";
    public static final String EvaluationResults2Directory = "./EvaluationResults2/";

    public static final String JavaAttentionalTranslation_Directory = "./Translations/JavaAttentionalTranslations/";
    public static final String PhpAttentionalTranslation_Directory = "./Translations/PhpAttentionalTranslations/";
    public static final String Voteshare_Directory = "./VoteShare/";
}
