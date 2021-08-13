import AttentionalTranslation.SkillTranslation;
import GoldenSet.ExpertUsers;
import Index.LuceneIndex;
import Utility.Constants;
import Utility.MAP;
import Utility.VoteShare;

import java.io.*;
import java.util.Date;

public class Main {
    public static void main(String[] args) {
        Date start = new Date();
        Main m = new Main();

        //Create a lucene index for specified subset of stackoverflow dataset
        m.IndexInputData("java");
        m.IndexInputData("php");

        // ASTM-1 and ASTM-2
        m.AttentionalSkillTranslation("java", "java_astm1_best", true, 10, "Word", "Test");
        m.AttentionalSkillTranslation("java", "java_astm2_best", false, 10, "Word", "Test");
        m.AttentionalSkillTranslation("php", "php_astm1_best", true, 10, "Word", "Test");
        m.AttentionalSkillTranslation("php", "php_astm2_best", false, 10, "Word", "Test");

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
     *
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

    public void AttentionalSkillTranslation(String tag, String model_name, boolean use_vote_share, int topWords, String type, String dataset) {
        if (tag.equals("java")) {
            String index_path = Constants.JavaIndexDirectory2;
            String IndexName = index_path.substring(index_path.lastIndexOf("/") + 1);

            SkillTranslation b = new SkillTranslation(index_path, "java", Constants.JavaXMLInput);

            String infile = Constants.JavaAttentionalTranslation_Directory + model_name + ".txt";
            File indir = new File(infile);
            if (!indir.exists())
                return;

            String outfile = model_name;
            if (use_vote_share)
                outfile += "_top" + topWords + "_voteshare";
            else
                outfile += "_top" + topWords + "_without_voteshare";
            System.out.println(infile);

            System.out.println("Translation Type: " + type);
            System.out.println("Dataset Part: " + dataset);

            String attentional_result_dir = outfile + "_" + IndexName;
            String attentional_result_filename = outfile + "_type" + type + "_dataset" + dataset;
            String DirName = Constants.Results_Directory + attentional_result_dir + "/";

            File dir = new File(Constants.Results_Directory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            b.blendOr(infile, Constants.Results_Directory + attentional_result_dir + "/" + attentional_result_filename, type, dataset, topWords, true, true, true, use_vote_share);// no cluster, without voteshare

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

            SkillTranslation b = new SkillTranslation(index_path, "php", Constants.PhpXMLInput);

            String infile = Constants.PhpAttentionalTranslation_Directory + model_name + ".txt";
            File indir = new File(infile);
            if (!indir.exists())
                return;

            String outfile = model_name;
            if (use_vote_share)
                outfile += "_top" + topWords + "_voteshare";
            else
                outfile += "_top" + topWords + "_without_voteshare";
            System.out.println(infile);

            System.out.println("Translation Type: " + type);
            System.out.println("Dataset Part: " + dataset);

            String attentional_result_dir = outfile + "_" + IndexName;
            String attentional_result_filename = outfile + "_type" + type + "_dataset" + dataset;
            String DirName = Constants.Results_Directory + attentional_result_dir + "/";

            File dir = new File(Constants.Results_Directory + attentional_result_dir);
            if (!dir.exists())
                dir.mkdirs();

            b.blendOr(infile, Constants.Results_Directory + attentional_result_dir + "/" + attentional_result_filename, type, dataset, topWords, true, true, true, use_vote_share);// no cluster, without voteshare

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
