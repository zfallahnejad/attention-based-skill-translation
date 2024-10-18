package AttentionalTranslation;

import Index.IndexUtility;
import Utility.Constants;
import Utility.Evaluator;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.jsoup.Jsoup;
import org.jsoup.select.Elements;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class SkillTranslation {
    String mainTag;
    String xml_input;
    String index_path;
    HashMap<Integer, Double> voteShare;
    HashMap<Integer, ArrayList<String>> post_tags;
    IndexUtility u;
    Analyzer analyzer;
    IndexReader reader;
    IndexSearcher searcher;

    private void loadVoteShare(String file_path) {
        //System.out.println("Loading voteshares");
        this.voteShare = new HashMap<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file_path));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split(",");
                if (line.trim().equals("aid,voteshare"))
                    continue;
                this.voteShare.put(Integer.parseInt(parts[0]), Double.parseDouble(parts[1]));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println("Loading voteshare done");
    }

    private void PostTags() {
        post_tags = new HashMap<>();
        System.out.println("Loading post tags started");

        File f = new File(Constants.Voteshare_Directory + mainTag + "/" + mainTag + "_posts_tags.csv");
        if (f.exists()) {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(Constants.Voteshare_Directory + mainTag + "/" + mainTag + "_posts_tags.csv"));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.trim().split(",");
                    Integer Id = Integer.parseInt(parts[0]);
                    ArrayList<String> Tags = new ArrayList<>();
                    String[] ss = parts[1].split("<|>");
                    for (String s : ss) {
                        if (s != null && s.trim().length() > 0)
                            Tags.add(s.trim());
                    }
                    post_tags.put(Id, Tags);
                }
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                BufferedReader reader = new BufferedReader(new FileReader(xml_input));
                PrintWriter out = new PrintWriter(Constants.Voteshare_Directory + mainTag + "/" + mainTag + "_posts_tags.csv");
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().startsWith("<row")) {
                        Elements row = Jsoup.parse(line).getElementsByTag("row");
                        Integer Id = Integer.parseInt(row.attr("Id"));
                        //Integer PostTypeId = Integer.parseInt(row.attr("PostTypeId"));
                        ArrayList<String> Tags = new ArrayList<>();
                        String[] ss = row.attr("Tags").split("<|>");
                        for (String s : ss) {
                            if (s != null && s.trim().length() > 0)
                                Tags.add(s.trim());
                        }
                        post_tags.put(Id, Tags);
                        out.println(Id + "," + row.attr("Tags"));
                    }
                }
                reader.close();
                out.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Loading post tags done!");
        System.out.println("Number of posts: " + post_tags.size());
    }

    private boolean hasTag(int aid, String tag) {
        return post_tags.get(aid).contains(tag);
    }

    public SkillTranslation(String indexPath, String topTag, String XMLInput, Boolean use_vote_share, String VoteshareInput) {
        try {
            mainTag = topTag;
            xml_input = XMLInput;
            index_path = indexPath;
            u = new IndexUtility(indexPath);
            reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
            searcher = new IndexSearcher(reader);
            analyzer = new StandardAnalyzer();
            PostTags();
            if (use_vote_share)
                loadVoteShare(VoteshareInput);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public HashMap<String, ArrayList<ProbTranslate>> loadTranslations(String infilePath, String TranslationType, String Dataset, int countWords) {
        HashMap<String, ArrayList<ProbTranslate>> tags = new HashMap<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(infilePath));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(", Dataset=");
                String type = parts[0].replace("Type=", "");
                if (!type.equalsIgnoreCase(TranslationType)) {
                    continue;
                }

                parts = parts[1].split(", Label=");
                String data_part = parts[0];
                if (!data_part.equalsIgnoreCase(Dataset))
                    continue;

                parts = parts[1].split(", Translations=");
                String tag = parts[0];
                String translations = parts[1].replace("[(", "").replace(")]", "");
                //if (tag.equals(mainTag))
                //    continue;

                ArrayList<ProbTranslate> e = new ArrayList<>();
                String[] tr_score = translations.split("\\), \\(");
                if (tr_score.length > 1) {
                    for (int i = 0; i < countWords; i++) {
                        String translate = tr_score[i].substring(1, tr_score[i].lastIndexOf(',') - 1);
                        String score = tr_score[i].substring(tr_score[i].lastIndexOf(',') + 2);
                        e.add(new ProbTranslate(translate, 1));
                        //e.add(new ProbTranslate(translate, score));
                    }
                }
                //else {
                //    // self translation if you don't have anny translation
                //    e.add(new ProbTranslate(tag, 1));
                //    e2.add(new ProbTranslate(tag, 1));
                //}
                tags.put(tag, e);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return tags;
    }

    public HashMap<String, ArrayList<ProbTranslate>> loadMultipleTranslations(ArrayList<String> infilePathList, String TranslationType, String Dataset, int countWords) {
        HashMap<String, ArrayList<ProbTranslate>> tags = new HashMap<>();
        try {
            for (String infilePath : infilePathList) {
                BufferedReader reader = new BufferedReader(new FileReader(infilePath));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(", Dataset=");
                    String type = parts[0].replace("Type=", "");
                    if (!type.equalsIgnoreCase(TranslationType)) {
                        continue;
                    }

                    parts = parts[1].split(", Label=");
                    String data_part = parts[0];
                    if (!data_part.equalsIgnoreCase(Dataset))
                        continue;

                    parts = parts[1].split(", Translations=");
                    String tag = parts[0];
                    String translations = parts[1].replace("[(", "").replace(")]", "");
                    //if (tag.equals(mainTag))
                    //    continue;

                    ArrayList<ProbTranslate> e = new ArrayList<>();
                    String[] tr_score = translations.split("\\), \\(");
                    if (tr_score.length > 1) {
                        for (int i = 0; i < countWords; i++) {
                            String translate = tr_score[i].substring(1, tr_score[i].lastIndexOf(',') - 1);
                            String score = tr_score[i].substring(tr_score[i].lastIndexOf(',') + 2);
                            e.add(new ProbTranslate(translate, 1));
                            //e.add(new ProbTranslate(translate, score));
                        }
                    }
                    //else {
                    //    // self translation if you don't have anny translation
                    //    e.add(new ProbTranslate(tag, 1));
                    //    e2.add(new ProbTranslate(tag, 1));
                    //}
                    if (tags.containsKey(tag)){
                        tags.get(tag).addAll(e);
                    } else{
                        tags.put(tag, e);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tags;
    }

    public HashMap<String, ArrayList<ProbTranslate>> loadTranslations_neshati(String infilePath, int countWords) {
        HashMap<String, ArrayList<ProbTranslate>> tags = new HashMap<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(infilePath));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] tgs = line.split("~");
                String tag = tgs[0];
                //if (tag.equals(mainTag))
                //    continue;

                ArrayList<ProbTranslate> e = new ArrayList<>();
                if (tgs.length > 1 && tgs[1] != null && tgs[1] != "") {
                    String[] trs = tgs[1].split(",");
                    for (int i = 0; i < countWords; i++) {
                        if (trs[i].contains(":")) {
                            String[] t = trs[i].split(":");
                            e.add(new ProbTranslate(t[0], 1));
                            //e.add(new ProbTranslate(t[0], Double.parseDouble(t[1])));
                        } else {
                            e.add(new ProbTranslate(trs[i], 1));
                        }
                    }
                }
                //else {
                //    // self translation if you don't have anny translation
                //    e.add(new ProbTranslate(tgs[0], 1));
                //}
                tags.put(tag, e);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return tags;
    }

    public void getUserTranslationScore(String tag, ArrayList<ProbTranslate> translations, boolean tagged, boolean selfTranslate, boolean answerOnly, boolean useVoteShare, String outfile, boolean have_phrase_translation) {
        // TODO why N=50000?
        HashMap<Integer, Double> UserScores = getTransaltionScoreOr(50000, translations, tag, tagged, selfTranslate, answerOnly, useVoteShare, have_phrase_translation);

        ValueComparator bvc = new ValueComparator(UserScores);
        TreeMap<Integer, Double> sorted_map = new TreeMap<Integer, Double>(bvc);
        sorted_map.putAll(UserScores);
        try {
            PrintWriter out = new PrintWriter(outfile);
            for (Map.Entry<Integer, Double> entry : sorted_map.entrySet()) {
                out.println(entry.getKey() + "," + entry.getValue());
            }
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Integer> loadUserTranslationScore(String outfile) {
        ArrayList<Integer> users = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(outfile));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                users.add(Integer.parseInt(parts[0]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return users;
    }

    public void blendOr(String infilePath, String outfilePath, String TranslationType, String Dataset, int countWords, boolean tagged, boolean selfTranslate, boolean answerOnly, boolean useVoteShare, boolean have_phrase_translation) {
        HashMap<String, ArrayList<ProbTranslate>> tags = loadTranslations(infilePath, TranslationType, Dataset, countWords);

        for (String tag : tags.keySet()) {
            System.out.println("tag: " + tag);
            String output_file = outfilePath + "_" + tag + ".txt";
            ArrayList<ProbTranslate> trans = tags.get(tag);
            if (trans.isEmpty()) {
                try {
                    PrintWriter out = new PrintWriter(output_file);
                    out.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                continue;
            }
            getUserTranslationScore(tag, trans, tagged, selfTranslate, answerOnly, useVoteShare, output_file, have_phrase_translation);
        }
    }

    public void blendOrMultiple(ArrayList<String> infilePathList, String outfilePath, String TranslationType, String Dataset, int countWords, boolean tagged, boolean selfTranslate, boolean answerOnly, boolean useVoteShare, boolean have_phrase_translation) {
        HashMap<String, ArrayList<ProbTranslate>> tags = loadMultipleTranslations(infilePathList, TranslationType, Dataset, countWords);

        for (String tag : tags.keySet()) {
            System.out.println(tag);
            String output_file = outfilePath + "_" + tag + ".txt";
            ArrayList<ProbTranslate> trans = tags.get(tag);
            if (trans.isEmpty()) {
                try {
                    PrintWriter out = new PrintWriter(output_file);
                    out.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
                continue;
            }
            getUserTranslationScore(tag, trans, tagged, selfTranslate, answerOnly, useVoteShare, output_file, have_phrase_translation);
        }
    }

    public void evaluate(String infilePath, String outfilePath, String TranslationType, String Dataset, int countWords, String GoldenSetDirectory, String evaluator_file) {
        HashMap<String, ArrayList<ProbTranslate>> tags = loadTranslations(infilePath, TranslationType, Dataset, countWords);

        try {
            PrintWriter out = new PrintWriter(evaluator_file);

            int cnt = 0;
            double sum = 0.0;
            for (String tag : tags.keySet()) {
                String output_file = outfilePath + "_" + tag + ".txt";
                ArrayList<Integer> lst = loadUserTranslationScore(output_file);
                if (lst.isEmpty()) {
                    System.out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + "," + tag +
                            ",0.0,0.0,0.0,0.0,");
                    out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + "," + tag +
                            ",0.0,0.0,0.0,0.0,");
                    continue;
                }
                Evaluator ev = new Evaluator();
                ArrayList<Integer> golden_list = getGoldenList(tag, GoldenSetDirectory);
                double map = ev.map(lst, golden_list);
                double p1 = ev.precisionAtK(lst, golden_list, 1);
                double p5 = ev.precisionAtK(lst, golden_list, 5);
                double p10 = ev.precisionAtK(lst, golden_list, 10);
                System.out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + "," + tag + "," +
                        map + "," + p1 + "," + p5 + "," + p10 + "," + lst.get(0));
                out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + "," + tag + "," +
                        map + "," + p1 + "," + p5 + "," + p10 + "," + lst.get(0));

                if (!Double.isNaN(map)) {
                    sum += map;
                    cnt++;
                }
            }
            if (!Double.isNaN(sum / cnt)) {
                System.out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + ",average," + (sum / cnt));
                out.println(TranslationType.toLowerCase() + "," + Dataset.toLowerCase() + ",average," + (sum / cnt));
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Integer> getGoldenList(String tag, String GoldenSetDirectory) {
        ArrayList<Integer> res = new ArrayList<>();
        //System.out.println("Loading golden list of " + tag + " started");
        try {
            BufferedReader reader = new BufferedReader(new FileReader(GoldenSetDirectory + tag + ".csv"));
            String line;
            while ((line = reader.readLine()) != null) {
                res.add(Integer.parseInt(line.trim().split(",")[0]));
            }
            reader.close();
            //System.out.println("Loading golden list of " + tag + "done!");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return res;
    }

    public HashMap<Integer, Double> getTransaltionScoreOr(Integer N, ArrayList<ProbTranslate> trans, String tag, boolean isTaged, boolean selfTranslate, boolean answerOnly, boolean useVoteShare, boolean phrase_translation) // throws IOException, ParseException
    {
        if (N == null) {
            N = 10000;
        }
        int hitsPerPage = N;
        Query q = null;
        for (ProbTranslate tran : trans) {
            if (q == null) {
                if (phrase_translation) {
                    q = u.SearchBodyForPhrase(tran.getWord());
                } else {
                    q = u.SearchBody(tran.getWord());
                }
            } else {
                if (phrase_translation) {
                    q = u.BooleanQueryOr(q, u.SearchBodyForPhrase(tran.getWord()));
                } else {
                    q = u.BooleanQueryOr(q, u.SearchBody(tran.getWord()));
                }
            }
        }
        if (answerOnly) {
            q = u.BooleanQueryAnd(q, u.SearchPostTypeID(2));
        }
        if (selfTranslate) {
            q = u.BooleanQueryOr(q, u.SearchBody(tag));
        }

        HashMap<Integer, Double> userScores = new HashMap<>();

        try {
            TopDocs results = searcher.search(q, 5 * hitsPerPage);
            ScoreDoc[] hits = results.scoreDocs;
            int numTotalHits = results.totalHits;
            //System.out.println("numTotalHits: " + numTotalHits);
            int start = 0;
            int end = Math.min(numTotalHits, hitsPerPage);
            for (int i = start; i < end; i++) {
                int docID = hits[i].doc;
                Document doc = searcher.doc(docID);
                if (isTaged && !hasTag(Integer.parseInt(doc.get("Id")), tag)) {
                    continue;
                }

                int uid = -1;
                try {
                    uid = Integer.parseInt(doc.get("OwnerUserId"));
                } catch (Exception ex) {
                    continue;
                }

                double score = 1.0;
                if (useVoteShare) {
                    int aid = Integer.parseInt(doc.get("Id"));
                    if (this.voteShare.containsKey(aid)) {
                        score = this.voteShare.get(aid);
                    } else {
                        score = 0;
                    }
                }
                if (userScores.containsKey(uid)) {
                    double oldScore = userScores.get(uid);
                    userScores.replace(uid, score + oldScore);
                } else {
                    userScores.put(uid, score);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return userScores;
    }

    class ProbTranslate {
        private String word;
        private double prob;

        public double getProb() {
            return prob;
        }

        public void setProb(double prob) {
            this.prob = prob;
        }

        public String getWord() {
            return word;
        }

        public void setWord(String word) {
            this.word = word;
        }

        public ProbTranslate(String word, double prob) {
            this.word = word;
            this.prob = prob;
        }
    }

    class ValueComparator implements Comparator<Integer> {
        Map<Integer, Double> base;

        public ValueComparator(Map<Integer, Double> base) {
            this.base = base;
        }

        // Note: this comparator imposes orderings that are inconsistent with equals.
        public int compare(Integer a, Integer b) {
            if (base.get(a) >= base.get(b)) {
                return -1;
            } else {
                return 1;
            } // returning 0 would merge keys
        }
    }
}
