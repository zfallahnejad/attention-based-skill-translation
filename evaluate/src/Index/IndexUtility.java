package Index;

import Utility.Constants;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.QueryBuilder;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashSet;

/**
 * Created by Zohreh on 6/18/2017.
 */
public class IndexUtility {
    IndexReader reader;
    IndexSearcher searcher;
    Analyzer analyzer;

    public IndexUtility(String index_dir) {
        try {
            reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_dir)));
            searcher = new IndexSearcher(reader);
            analyzer = new StandardAnalyzer();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Query SearchPostId(Integer PostID) {
        Query query = NumericRangeQuery.newIntRange("Id", PostID, PostID, true, true);
        return query;
    }

    public Query SearchTag(String tag) {
        Query query = new TermQuery(new Term("Tags", tag));
        return query;
    }

    public Query SearchTopic(Integer topic) {
        Query query = NumericRangeQuery.newIntRange("Topics", topic, topic, true, true);
        return query;
    }

    public Query SearchBody(String word) {
        Query query = new TermQuery(new Term("Body", word));
        // Query query = new QueryParser("Body", analyzer).parse(word);
        return query;
    }

    public Query SearchTitle(String word) {
        Query query = new TermQuery(new Term("Title", word));
        return query;
    }

    public Query SearchOwnerUserId(Integer UsersID) {
        Query query = NumericRangeQuery.newIntRange("OwnerUserId", UsersID, UsersID, true, true);
        return query;
    }

    public Query SearchPostTypeID(Integer PostTypeId) {
        Query query = NumericRangeQuery.newIntRange("PostTypeId", PostTypeId, PostTypeId, true, true);
        return query;
    }

    public Query SearchAcceptedAnswerId(Integer AcceptedAnswerId) {
        Query query = NumericRangeQuery.newIntRange("AcceptedAnswerId", AcceptedAnswerId, AcceptedAnswerId, true, true);
        return query;
    }

    public Query SearchCreationDateRange(Calendar c1, Calendar c2) {
        return NumericRangeQuery.newLongRange("CreationDate", c1.getTimeInMillis(), c2.getTimeInMillis(), true, true);
    }

    public Query SearchCreationDate(int year) {
        Calendar c1 = getFirstDay(year);
        Calendar c2 = getLastDay(year);
        return NumericRangeQuery.newLongRange("CreationDate", c1.getTimeInMillis(), c2.getTimeInMillis(), true, true);
    }

    public Query SearchCreationDateRange(int year1, int year2) {
        Calendar c1 = getFirstDay(year1);
        Calendar c2 = getLastDay(year2);
        return NumericRangeQuery.newLongRange("CreationDate", c1.getTimeInMillis(), c2.getTimeInMillis(), true, true);
    }

    private Calendar getFirstDay(int year) {
        Calendar c1 = Calendar.getInstance();
        c1.set(year, Calendar.JANUARY, 1, 0, 0, 0);
        c1.clear(Calendar.MINUTE);
        c1.clear(Calendar.HOUR);
        c1.clear(Calendar.SECOND);
        c1.clear(Calendar.MILLISECOND);
        return c1;
    }

    private Calendar getLastDay(int year) {
        Calendar c2 = Calendar.getInstance();
        c2.set(year, Calendar.DECEMBER, 31, 23, 59);
        c2.clear(Calendar.MINUTE);
        c2.clear(Calendar.HOUR);
        c2.clear(Calendar.SECOND);
        c2.clear(Calendar.MILLISECOND);
        return c2;
    }

    public BooleanQuery BooleanQueryOr(Query q1, Query q2) {
        BooleanQuery query = new BooleanQuery();
        query.add(q1, BooleanClause.Occur.SHOULD);
        query.add(q2, BooleanClause.Occur.SHOULD);
        return query;
    }

    public BooleanQuery BooleanQueryAnd(Query q1, Query q2) {
        BooleanQuery query = new BooleanQuery();
        query.add(q1, BooleanClause.Occur.MUST);
        query.add(q2, BooleanClause.Occur.MUST);
        return query;
    }

    /**
     * Get PostID of query result
     * @param q input query
     * @return Set of PostIDs
     */
    public HashSet<Integer> getPostIDs(Query q) {
        try {
            HashSet<Integer> PIDs = new HashSet<Integer>();
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits + " total matching documents");

            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                //System.out.println(d.toString());
                PIDs.add(Integer.parseInt(d.get("Id")));
            }
            return PIDs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void getPost(Query q) {
        try {
            ArrayList<Integer> PIDs = new ArrayList<Integer>();
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);

            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                System.out.println(d.toString());
            }
            return;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return;
    }

    public ArrayList<Integer> getAcceptedAnswerId(Query q) {
        try {
            ArrayList<Integer> acceptedAnswerIds = new ArrayList<Integer>();
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits + " total matching documents");

            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                //System.out.println(d.toString());
                System.out.println("AcceptedAnswerId:" + d.get("AcceptedAnswerId"));
                acceptedAnswerIds.add(Integer.parseInt(d.get("AcceptedAnswerId")));
            }
            return acceptedAnswerIds;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public Integer getDocCount(Query q) {
        try {
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            return hits.totalHits;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -1;
    }

    public ArrayList<Integer> getActivityYearsByExpertID(Integer eid) {
        ArrayList<Integer> activityYears = new ArrayList<Integer>();
        try {
            for (int year = 2008; year < 2016; year++) {
                Query q = BooleanQueryAnd(SearchOwnerUserId(eid), SearchCreationDate(year));
                TopDocs hits = searcher.search(q, 1);//retrieve more than one docs
                //System.out.println(hits.totalHits + " total matching documents");
                if (hits.totalHits > 0) {
                    activityYears.add(year);
                }
            }
            return activityYears;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return activityYears;
    }

    public ArrayList<Integer> getActivityYearsByExpertID(Integer eid, Integer futureYear) {
        ArrayList<Integer> activityYears = new ArrayList<Integer>();
        try {
            for (int y = 2008; y < futureYear; y++) {
                Query q = BooleanQueryAnd(SearchOwnerUserId(eid), SearchCreationDate(y));
                TopDocs hits = searcher.search(q, 1);//retrieve more than one docs
                //System.out.println(hits.totalHits + " total matching documents");
                if (hits.totalHits > 0) {
                    activityYears.add(y);
                }
            }
            return activityYears;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return activityYears;
    }

    public long getFreqOfWordInBody(String word, String field) {
        try {
            QueryBuilder builder = new QueryBuilder(analyzer);
            Query query = builder.createBooleanQuery(field, word);
            TopDocs hits = searcher.search(query, Integer.MAX_VALUE);
            System.out.println(hits.totalHits + " total matching documents");
            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                Terms terms = reader.getTermVector(docId, field); //get terms vectors for one document and one field
                if (terms != null && terms.size() > 0) {
                    TermsEnum termsEnum = terms.iterator(); // access the terms for this field
                    BytesRef term = null;
                    while ((term = termsEnum.next()) != null) {
                        final String keyword = term.utf8ToString();
                        long termFreq = termsEnum.totalTermFreq();
                        if (keyword.equalsIgnoreCase(word))
                            return termFreq;
                        //System.out.println("DocID: " + d.get("Id") + ", term: " + keyword + ", termFreq = " + termFreq);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0;
    }

    public Integer getOwnerUserId(Integer PostID) {
        try {
            Query q = SearchPostId(PostID);
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits + " total matching documents");

            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                //System.out.println(d.toString());
                if (Integer.parseInt(d.get("Id")) == PostID)
                    return Integer.parseInt(d.get("OwnerUserId"));
            }
            return -2;//not found
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -2;
    }

    public HashSet<Integer> getExpertsBYTagandYear(String Tag, Integer year) {
        Query q = BooleanQueryAnd(SearchCreationDate(year), SearchTag(Tag));
        HashSet<Integer> ExpertIDs = new HashSet<Integer>();
        try {
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits+" total matching documents");
            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                ExpertIDs.add(Integer.parseInt(d.get("OwnerUserId")));
            }
            return ExpertIDs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ExpertIDs;
    }

    public HashSet<Integer> getExpertsBYWordandYear(String Tag, Integer year) {
        Query q = BooleanQueryAnd(SearchCreationDate(year), SearchBody(Tag));
        HashSet<Integer> ExpertIDs = new HashSet<Integer>();
        try {
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits+" total matching documents");
            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                ExpertIDs.add(Integer.parseInt(d.get("OwnerUserId")));
            }
            return ExpertIDs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ExpertIDs;
    }

    public HashSet<String> getTags(Query q) {
        HashSet<String> Tags = new HashSet<String>();
        try {
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                for (IndexableField tag : d.getFields("Tags")) {
                    //if (!Tags.contains(tag.stringValue()))
                    Tags.add(tag.stringValue());
                    if (tag.stringValue().equalsIgnoreCase(""))
                        System.out.println("Errrrrrrrrorrrrrrrrrrrrrrrr!!!!!!!!!!!!!!!!!!!");
                }
            }
            return Tags;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return Tags;
    }

    public HashSet<Integer> getExpertsBYTopicandYear(Query q) {
        HashSet<Integer> ExpertIDs = new HashSet<Integer>();
        try {
            TopDocs hits = searcher.search(q, Integer.MAX_VALUE);
            //System.out.println(hits.totalHits+" total matching documents");
            ScoreDoc[] ScDocs = hits.scoreDocs;
            for (int i = 0; i < ScDocs.length; ++i) {
                int docId = ScDocs[i].doc;
                Document d = searcher.doc(docId);
                ExpertIDs.add(Integer.parseInt(d.get("OwnerUserId")));
            }
            return ExpertIDs;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ExpertIDs;
    }

    // just for test!
    public static void main(String args[]) {
        //IndexUtility u = new IndexUtility(Constants.JavaIndexDirectory2);
        //u.getPost(u.SearchOwnerUserId(330086));
        //HashSet<Integer> posts = u.getPostIDs(u.BooleanQueryAnd(u.SearchBody("httpclient"), u.SearchTag("android")));
        //System.out.println(posts.toString());
        //System.out.println(u.getDocCount(u.BooleanQueryAnd(u.SearchCreationDate(2008), u.SearchTopic(0))));//196
        //System.out.println(u.getDocCount(u.BooleanQueryAnd(u.SearchOwnerUserId(1793), u.BooleanQueryAnd(u.SearchCreationDate(2008), u.SearchTopic(0)))));//1

        //System.out.println(u.getDocCount(u.BooleanQueryAnd(u.SearchPostTypeID(2),u.BooleanQueryAnd(u.SearchCreationDate(2008), u.SearchTopic(0)))));//130
        //System.out.println(u.getDocCount(u.BooleanQueryAnd(u.SearchPostTypeID(2),u.BooleanQueryAnd(u.SearchOwnerUserId(1793), u.BooleanQueryAnd(u.SearchCreationDate(2008), u.SearchTopic(0))))));//0
    }
}
