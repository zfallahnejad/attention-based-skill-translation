package GoldenSet;

import Index.IndexUtility;
import Utility.Constants;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class ExpertUsers {
    IndexUtility u;
    IndexSearcher searcher;
    IndexReader reader;
    HashSet<Integer> AcceptedAnswers;
    HashMap<Integer, Integer> User_NumAcceptedAnswers;
    HashMap<Integer, Integer> User_NumAnswers;
    Integer AcceptedAnswerThreshold;

    public ExpertUsers(String index_path, Integer accepted_answer_threshold) {
        try {
            u = new IndexUtility(index_path);
            reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_path)));
            searcher = new IndexSearcher(reader);
            AcceptedAnswers = new HashSet<Integer>();
            User_NumAcceptedAnswers = new HashMap<Integer, Integer>();
            User_NumAnswers = new HashMap<Integer, Integer>();
            AcceptedAnswerThreshold = accepted_answer_threshold;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void FindExperts(String[] top_tags, String GoldenSetDirectory) {
        File dir = new File(GoldenSetDirectory);
        if (!dir.exists())
            dir.mkdirs();

        SearchForAcceptedAnswers();
        for (String tag : top_tags) {
            System.out.println(tag);
            try {
                PrintWriter out = new PrintWriter(GoldenSetDirectory + tag + ".csv");
                SearchForUserAnswers(tag);
                for (Integer user : User_NumAcceptedAnswers.keySet()) {
                    if (user == -1)
                        continue;
                    if ((User_NumAcceptedAnswers.get(user) > AcceptedAnswerThreshold) && ((User_NumAcceptedAnswers.get(user) * 1.0 / User_NumAnswers.get(user)) > 0.4)) {
                        System.out.println(user + "," + tag);
                        out.println(user + "," + tag);
                    }
                }
                User_NumAcceptedAnswers.clear();
                User_NumAnswers.clear();
                out.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    private void SearchForAcceptedAnswers() {
        try {
            Query q = u.SearchPostTypeID(1);
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            ScoreDoc[] ScDocs = hits.scoreDocs;

            for (int i = 0; i < ScDocs.length; i++) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                try {
                    AcceptedAnswers.add(Integer.parseInt(d.get("AcceptedAnswerId")));
                } catch (Exception e) {
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void SearchForUserAnswers(String tag) {
        try {
            Query q = u.BooleanQueryAnd(u.SearchPostTypeID(2), u.SearchTag(tag));
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            ScoreDoc[] ScDocs = hits.scoreDocs;

            for (int i = 0; i < ScDocs.length; i++) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                try {
                    Integer oid = Integer.parseInt(d.get("OwnerUserId"));
                    if (User_NumAnswers.containsKey(oid)) {
                        User_NumAnswers.put(oid, User_NumAnswers.get(oid) + 1);
                    } else {
                        User_NumAnswers.put(oid, 1);
                        User_NumAcceptedAnswers.put(oid, 0);
                    }
                    if (AcceptedAnswers.contains(Integer.parseInt(d.get("Id")))) {
                        User_NumAcceptedAnswers.put(oid, User_NumAcceptedAnswers.get(oid) + 1);
                    }
                } catch (Exception e) {
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        //ExpertUsers b = new ExpertUsers(Constants.JavaIndexDirectory, 10);
        //b.FindExperts(Constants.Java_TopTags2, Constants.JavaGoldenSetDirectory);

        ExpertUsers e = new ExpertUsers(Constants.PhpIndexDirectory2, 6);
        e.FindExperts(Constants.Php_TopTags2, Constants.PhpGoldenSetDirectory);
    }
}
