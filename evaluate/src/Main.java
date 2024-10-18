import AttentionalTranslation.SkillTranslation;
import GoldenSet.ExpertUsers;
import Index.LuceneIndex;
import Utility.*;

import java.io.*;
import java.util.Date;

public class Main {
    public static void main(String[] args) {
        String outfile_prefix;
        String voteshare_path;

        Date start = new Date();
        Main m = new Main();

        //Create a lucene index for specified subset of stackoverflow dataset
        //m.IndexInputData("java");
        //m.IndexInputData("php");

        //VoteShare v = new VoteShare();
        //v.compute_voteshare("java", Constants.JavaXMLInput);
        //v.compute_voteshare("php", Constants.PhpXMLInput);
        //VoteShare2 v = new VoteShare2();
        //v.compute_voteshare("java", Constants.JavaXMLInput);
        //v.compute_voteshare("php", Constants.PhpXMLInput);
        //VoteShare3 v = new VoteShare3();
        //v.compute_voteshare("java", Constants.JavaXMLInput);
        //v.compute_voteshare("php", Constants.PhpXMLInput);
        //VoteShare4 v = new VoteShare4();
        //v.compute_voteshare("java", Constants.JavaXMLInput);
        //v.compute_voteshare("php", Constants.PhpXMLInput);
        //VoteShare5 v = new VoteShare5();
        //v.compute_voteshare("java", Constants.JavaXMLInput);
        //v.compute_voteshare("php", Constants.PhpXMLInput);

        // ASTM-1 and ASTM-2 - binary scoring
        //outfile_prefix = "java_astm1_best" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm1_best", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "java_astm2_best" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm2_best", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm1_best" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm1_best", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm2_best" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm2_best", false, 10, "Word", "Test", "", outfile_prefix);

        // ASTM-1 and ASTM-2 - voteshare scoring - nobari
        //outfile_prefix = "java_astm1_best" + "_top" + 10 + "_voteshare_nobari";
        //voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //m.AttentionalSkillTranslation("java", "java_astm1_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "java_astm2_best" + "_top" + 10 + "_voteshare_nobari";
        //voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //m.AttentionalSkillTranslation("java", "java_astm2_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "php_astm1_best" + "_top" + 10 + "_voteshare_nobari";
        //voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //m.AttentionalSkillTranslation("php", "php_astm1_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "php_astm2_best" + "_top" + 10 + "_voteshare_nobari";
        //voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //m.AttentionalSkillTranslation("php", "php_astm2_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);

        //for (int v = 1; v < 7; v++) {
        //    // ASTM-1 and ASTM-2 - voteshare scoring - my voteshare version v
        //    outfile_prefix = "java_astm1_best" + "_top" + 10 + "_voteshare_v" + v;
        //    voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_v" + v + ".csv";
        //    m.AttentionalSkillTranslation("java", "java_astm1_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "java_astm2_best" + "_top" + 10 + "_voteshare_v" + v;
        //    voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_v" + v + ".csv";
        //    m.AttentionalSkillTranslation("java", "java_astm2_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "php_astm1_best" + "_top" + 10 + "_voteshare_v" + v;
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_v" + v + ".csv";
        //    m.AttentionalSkillTranslation("php", "php_astm1_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "php_astm2_best" + "_top" + 10 + "_voteshare_v" + v;
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_v" + v + ".csv";
        //    m.AttentionalSkillTranslation("php", "php_astm2_best", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //}

        //for (int t = 1; t < 5; t++) {
        //    int top = 2 * t;
        //    // ASTM-1 and ASTM-2 - binary scoring
        //
        //    outfile_prefix = "java_astm1_best" + "_top" + top + "_without_voteshare";
        //    m.AttentionalSkillTranslation("java", "java_astm1_best", false, top, "Word", "Test", "", outfile_prefix);
        //    outfile_prefix = "java_astm2_best" + "_top" + top + "_without_voteshare";
        //    m.AttentionalSkillTranslation("java", "java_astm2_best", false, top, "Word", "Test", "", outfile_prefix);
        //    outfile_prefix = "php_astm1_best" + "_top" + top + "_without_voteshare";
        //    m.AttentionalSkillTranslation("php", "php_astm1_best", false, top, "Word", "Test", "", outfile_prefix);
        //    outfile_prefix = "php_astm2_best" + "_top" + top + "_without_voteshare";
        //    m.AttentionalSkillTranslation("php", "php_astm2_best", false, top, "Word", "Test", "", outfile_prefix);
        //
        //    // ASTM-1 and ASTM-2 - voteshare scoring - nobari
        //    outfile_prefix = "java_astm1_best" + "_top" + top + "_voteshare_nobari";
        //    voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //    m.AttentionalSkillTranslation("java", "java_astm1_best", true, top, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "java_astm2_best" + "_top" + top + "_voteshare_nobari";
        //    voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //    m.AttentionalSkillTranslation("java", "java_astm2_best", true, top, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "php_astm1_best" + "_top" + top + "_voteshare_nobari";
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //    m.AttentionalSkillTranslation("php", "php_astm1_best", true, top, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "php_astm2_best" + "_top" + top + "_voteshare_nobari";
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //    m.AttentionalSkillTranslation("php", "php_astm2_best", true, top, "Word", "Test", voteshare_path, outfile_prefix);
        //}

        //for (int wd = 1; wd < 5; wd++) {
        //    if (wd == 2)
        //        continue;
        //    int word_dim = 50 * wd;
        //    System.out.println(word_dim);
        //
        //    // ASTM-1 and ASTM-2 - binary scoring
        //    outfile_prefix = "java_astm1_wd" + word_dim + "_top" + 10 + "_without_voteshare";
        //    m.AttentionalSkillTranslation("java", "java_astm1_wd" + word_dim, false, 10, "Word", "Test", "", outfile_prefix);
        //    outfile_prefix = "java_astm2_wd" + word_dim + "_top" + 10 + "_without_voteshare";
        //    m.AttentionalSkillTranslation("java", "java_astm2_wd" + word_dim, false, 10, "Word", "Test", "", outfile_prefix);
        //    outfile_prefix = "php_astm1_wd" + word_dim + "_top" + 10 + "_without_voteshare";
        //    m.AttentionalSkillTranslation("php", "php_astm1_wd" + word_dim, false, 10, "Word", "Test", "", outfile_prefix);
        //
        //    // ASTM-1 and ASTM-2 - voteshare scoring - nobari
        //    voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //    outfile_prefix = "java_astm1_wd" + word_dim + "_top" + 10 + "_voteshare_nobari";
        //    m.AttentionalSkillTranslation("java", "java_astm1_wd" + word_dim, true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //    outfile_prefix = "java_astm2_wd" + word_dim + "_top" + 10 + "_voteshare_nobari";
        //    m.AttentionalSkillTranslation("java", "java_astm2_wd" + word_dim, true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //    outfile_prefix = "php_astm1_wd" + word_dim + "_top" + 10 + "_voteshare_nobari";
        //    m.AttentionalSkillTranslation("php", "php_astm1_wd" + word_dim, true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //}

        //for (int wd = 2; wd < 5; wd++) {
        //    int word_dim = 50 * wd;
        //    System.out.println(word_dim);
        //
        //    // ASTM-1 and ASTM-2 - binary scoring
        //    outfile_prefix = "php_astm2_wd" + word_dim + "_top" + 10 + "_without_voteshare";
        //    m.AttentionalSkillTranslation("php", "php_astm2_wd" + word_dim, false, 10, "Word", "Test", "", outfile_prefix);
        //
        //    // ASTM-1 and ASTM-2 - voteshare scoring - nobari
        //    voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //    outfile_prefix = "php_astm2_wd" + word_dim + "_top" + 10 + "_voteshare_nobari";
        //    m.AttentionalSkillTranslation("php", "php_astm2_wd" + word_dim, true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //}

        //outfile_prefix = "java_astm1_dp05" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm1_dp05", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "java_astm1_dp075" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm1_dp075", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "java_astm2_dp05" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm2_dp05", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "java_astm2_dp1" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("java", "java_astm2_dp1", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm1_dp05" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm1_dp05", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm1_dp075" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm1_dp075", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm2_dp05" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm2_dp05", false, 10, "Word", "Test", "", outfile_prefix);
        //outfile_prefix = "php_astm2_dp075" + "_top" + 10 + "_without_voteshare";
        //m.AttentionalSkillTranslation("php", "php_astm2_dp075", false, 10, "Word", "Test", "", outfile_prefix);

        //voteshare_path = Constants.Voteshare_Directory + "java" + "/" + "java_vote_share_nobari.csv";
        //outfile_prefix = "java_astm1_dp05" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("java", "java_astm1_dp05", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "java_astm1_dp075" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("java", "java_astm1_dp075", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "java_astm2_dp05" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("java", "java_astm2_dp05", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "java_astm2_dp1" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("java", "java_astm2_dp1", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //voteshare_path = Constants.Voteshare_Directory + "php" + "/" + "php_vote_share_nobari.csv";
        //outfile_prefix = "php_astm1_dp05" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("php", "php_astm1_dp05", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "php_astm1_dp075" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("php", "php_astm1_dp075", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "php_astm2_dp05" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("php", "php_astm2_dp05", true, 10, "Word", "Test", voteshare_path, outfile_prefix);
        //outfile_prefix = "php_astm2_dp075" + "_top" + 10 + "_voteshare_nobari";
        //m.AttentionalSkillTranslation("php", "php_astm2_dp075", true, 10, "Word", "Test", voteshare_path, outfile_prefix);

        Date end = new Date();
        System.out.println(end.getTime() - start.getTime() + " total milliseconds");
    }

    /**
     * Construct Lucene Index from xml input data
     * We build two version of our indexes. One of them used LMJelinekMercerSimilarity and the other one doesn't.
     */
    public void IndexInputData(String tag) {
        LuceneIndex l = new LuceneIndex();

        if (tag.equals("java")) {
            l.setUp(Constants.JavaIndexDirectory2);
            l.index(Constants.JavaXMLInput);
        }

        if (tag.equals("php")) {
            l.setUp(Constants.PhpIndexDirectory2);
            l.index(Constants.PhpXMLInput);
        }
    }

    /**
     * Finds Experts Users to be used as golden set
     * This function considered users expertise in each tag
     * <p>
     * Note It seems that this implementation is a not equal to their implementation of golden set creator so don't use it
     */
    public void FindExpertUsers(String tag) {
        if (tag.equals("java")) {
            ExpertUsers e = new ExpertUsers(Constants.JavaIndexDirectory2, 10);
            e.FindExperts(Constants.Java_TopTags2, Constants.JavaGoldenSetDirectory);
        }

        if (tag.equals("php")) {
            ExpertUsers e = new ExpertUsers(Constants.PhpIndexDirectory2, 6);
            e.FindExperts(Constants.Php_TopTags2, Constants.PhpGoldenSetDirectory);
        }
    }

    public void AttentionalSkillTranslation(String tag, String model_name, boolean use_vote_share, int topWords, String type, String dataset, String voteshare_file, String outfile_prefix) {
        if (tag.equals("java")) {
            String index_path = Constants.JavaIndexDirectory2;
            String IndexName = index_path.substring(index_path.lastIndexOf("/") + 1);

            SkillTranslation b = new SkillTranslation(index_path, "java", Constants.JavaXMLInput, use_vote_share, voteshare_file);

            String infile = Constants.JavaAttentionalTranslation_Directory + model_name + ".txt";
            File indir = new File(infile);
            if (!indir.exists())
                return;

            System.out.println(infile);
            System.out.println("Translation Type: " + type);
            System.out.println("Dataset Part: " + dataset);
            System.out.println("Use vote_share: " + use_vote_share);

            String attentional_result_dir = outfile_prefix + "_" + IndexName;
            String attentional_result_filename = outfile_prefix + "_type" + type + "_dataset" + dataset;
            String DirName = Constants.Results_Directory + attentional_result_dir + "/";

            File dir = new File(Constants.Results_Directory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            b.blendOr(infile, Constants.Results_Directory + attentional_result_dir + "/" + attentional_result_filename, type, dataset, topWords, true, true, true, use_vote_share, false);// no cluster, without voteshare

            dir = new File(Constants.EvaluationResultsDirectory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            //getMAP(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getMAP(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Python, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Python, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Python, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Python, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            getMAP(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Neshati, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Java_TopTags3, Constants.JavaGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
        } else if (tag.equals("php")) {
            String index_path = Constants.PhpIndexDirectory2;
            String IndexName = index_path.substring(index_path.lastIndexOf("/") + 1);

            SkillTranslation b = new SkillTranslation(index_path, "php", Constants.PhpXMLInput, use_vote_share, voteshare_file);

            String infile = Constants.PhpAttentionalTranslation_Directory + model_name + ".txt";
            File indir = new File(infile);
            if (!indir.exists())
                return;

            System.out.println(infile);
            System.out.println("Translation Type: " + type);
            System.out.println("Dataset Part: " + dataset);
            System.out.println("Use vote_share: " + use_vote_share);

            String attentional_result_dir = outfile_prefix + "_" + IndexName;
            String attentional_result_filename = outfile_prefix + "_type" + type + "_dataset" + dataset;
            String DirName = Constants.Results_Directory + attentional_result_dir + "/";

            File dir = new File(Constants.Results_Directory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            b.blendOr(infile, Constants.Results_Directory + attentional_result_dir + "/" + attentional_result_filename, type, dataset, topWords, true, true, true, use_vote_share, false);// no cluster, without voteshare

            dir = new File(Constants.EvaluationResultsDirectory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            //getMAP(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_JavaGolden");
            //getMAP(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Python, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Python, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Python, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            //getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Python, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_PythonGolden");
            getMAP(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Neshati, DirName, attentional_result_filename, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 1, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 5, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
            getPat(Constants.Php_TopTags3, Constants.PhpGoldenSetDirectory_Neshati, DirName, attentional_result_filename, 10, "txt", attentional_result_dir, attentional_result_filename + "_NeshatiGolden");
        }
    }

    /**
     * This function evaluate results based on MAP mteric
     */
    private void getMAP(String[] TopTags, String GoldenSetDirectory, String DirName, String fileName, String
            format, String output_dir, String output_name) {
        File directory = new File(Constants.EvaluationResultsDirectory);
        if (!directory.exists()) {
            directory.mkdir();
        }
        directory = new File(Constants.EvaluationResultsDirectory + output_dir);
        if (!directory.exists()) {
            directory.mkdir();
        }

        MAP m = new MAP();
        try {
            PrintWriter out = new PrintWriter(Constants.EvaluationResultsDirectory + output_dir + "/" + output_name + "_MAP.csv");

            for (String tag : TopTags) {
                //System.out.println(tag);
                String map_result = m.computeMAP(tag, GoldenSetDirectory, DirName, fileName, format);
                out.println(map_result);
            }

            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * This function evaluate results based on P@n mteric
     */
    private void getPat(String[] TopTags, String GoldenSetDirectory, String DirName, String fileName,
                        int n, String format, String output_dir, String output_name) {
        File directory = new File(Constants.EvaluationResultsDirectory + output_dir);
        if (!directory.exists()) {
            directory.mkdir();
        }

        MAP m = new MAP();
        try {
            PrintWriter out = new PrintWriter(Constants.EvaluationResultsDirectory + output_dir + "/" + output_name + "_P_" + n + ".csv");

            for (String tag : TopTags) {
                //System.out.println(tag);
                String pat_result = m.computePat(tag, GoldenSetDirectory, DirName, fileName, n, format);
                out.println(pat_result);
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
