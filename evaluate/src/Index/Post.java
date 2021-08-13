package Index;

import org.jsoup.Jsoup;
import org.jsoup.select.Elements;
import org.apache.lucene.document.*;

import javax.xml.datatype.DatatypeFactory;
import javax.xml.datatype.XMLGregorianCalendar;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;

public class Post {
    SimpleDateFormat formatter;
    public Integer Id;
    public Integer PostTypeId;
    public Integer ParentId;
    public Integer AcceptedAnswerId;
    public Date CreationDate;
    public Integer Score;
    public Integer ViewCount;
    public String Body;
    public String Code;
    public Integer OwnerUserId;
    public String OwnerDisplayName;
    public Integer LastEditorUserId;
    public String LastEditorDisplayName;
    public Date LastEditDate;
    public Date LastActivityDate;
    public Date ClosedDate;
    public String Title;
    public ArrayList<String> Tags;
    public ArrayList<Integer> Topics;
    public Integer AnswerCount;
    public Integer CommentCount;
    public Integer FavoriteCount;
    public Date CommunityOwnedDate;

    public Post(String xmlLine) {
        formatter = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS");

        Elements row = Jsoup.parse(xmlLine).getElementsByTag("row");
        Id = getIntegerValue(row, "Id");
        PostTypeId = getIntegerValue(row, "PostTypeId");
        ParentId = getIntegerValue(row, "ParentId");
        AcceptedAnswerId = getIntegerValue(row, "AcceptedAnswerId");
        CreationDate = getDateValue(row, "CreationDate");
        Score = getIntegerValue(row, "Score");
        ViewCount = getIntegerValue(row, "ViewCount");
        Body = getStringValue(row, "Body");
        Code = generateCode(Body);
        OwnerUserId = getIntegerValue(row, "OwnerUserId");
        OwnerDisplayName = getStringValue(row, "OwnerDisplayName");
        LastEditorUserId = getIntegerValue(row, "LastEditorUserId");
        LastEditorDisplayName = getStringValue(row, "LastEditorDisplayName");
        LastEditDate = getDateValue(row, "LastEditDate");
        LastActivityDate = getDateValue(row, "LastActivityDate");
        ClosedDate = getDateValue(row, "ClosedDate");
        Title = getStringValue(row, "Title");
        Tags = getStringList(row, "Tags");
        Topics = getTopicSet(row);
        AnswerCount = getIntegerValue(row, "AnswerCount");
        CommentCount = getIntegerValue(row, "CommentCount");
        FavoriteCount = getIntegerValue(row, "FavoriteCount");
        CommunityOwnedDate = getDateValue(row, "CommunityOwnedDate");
    }

    private String generateCode(String Body) {
        org.jsoup.nodes.Document doc = Jsoup.parse(Body);
        Elements paragraphs = doc.select("code");
        StringBuilder sb = new StringBuilder();
        for (org.jsoup.nodes.Element p : paragraphs) {
            sb.append(p.text()).append(" ");
        }
        return sb.toString();
    }

    private ArrayList<String> getStringList(Elements row, String tag) {
        ArrayList<String> out = new ArrayList<>();
        String[] ss = row.attr(tag).split("<|>");
        for (String s : ss) {
            if (s != null && s.trim().length() > 0)
                out.add(s.trim());
        }
        return out;
    }

    private ArrayList<Integer> getTopicSet(Elements row) {
        ArrayList<Integer> out = new ArrayList<Integer>();
        String[] ss = row.attr("Topics").split("#");
        for (String s : ss) {
            if (s != null && s.trim().length() > 0)
                out.add(Integer.parseInt(s.trim()));
        }
        return out;
    }

    private String getStringValue(Elements row, String tag) {
        try {
            String html = row.attr(tag);
            html = html.replace("&lt;", "<").replace("&gt;", ">");
            return Jsoup.parse(html).text();
        } catch (Exception e) {
            return null;
        }
    }

    private Date getDateValue(Elements row, String tag) {
        Date date;
        try {
            XMLGregorianCalendar cal = DatatypeFactory.newInstance().newXMLGregorianCalendar(row.attr(tag));
            Calendar c3 = cal.toGregorianCalendar();
            c3.clear(Calendar.MINUTE);
            c3.clear(Calendar.HOUR);
            c3.clear(Calendar.SECOND);
            c3.clear(Calendar.MILLISECOND);
            date = c3.getTime();
        } catch (Exception e) {
            date = null;
        }
        return date;
    }

    private Integer getIntegerValue(Elements row, String tag) {
        Integer value;
        try {
            value = Integer.parseInt(row.attr(tag));
        } catch (Exception e) {
            value = null;
        }
        return value;
    }

    public static FieldType getVectorField() {
        FieldType myFieldType = new FieldType(TextField.TYPE_STORED);
        myFieldType.setStoreTermVectors(true);
        return myFieldType;
    }

    public Document getLuceneDocument() {
        Document doc = new Document();

        if (Id != null)
            doc.add(new IntField("Id", Id, Field.Store.YES));

        if (PostTypeId != null)
            doc.add(new IntField("PostTypeId", PostTypeId, Field.Store.YES));

        if (ParentId != null)
            doc.add(new IntField("ParentId", ParentId, Field.Store.YES));

        if (AcceptedAnswerId != null)
            doc.add(new IntField("AcceptedAnswerId", AcceptedAnswerId, Field.Store.YES));

        if (CreationDate != null)
            doc.add(new LongField("CreationDate", CreationDate.getTime(), Field.Store.YES));

        if (Score != null) {
            doc.add(new IntField("Score", Score, Field.Store.YES));
            doc.add(new SortedNumericDocValuesField("SortScore", Score));
        }

        if (ViewCount != null) {
            doc.add(new IntField("ViewCount", ViewCount, Field.Store.YES));
            doc.add(new SortedNumericDocValuesField("SortViewCount", ViewCount));
        }

        if (Body != null) {
            doc.add(new Field("Body", Body, getVectorField()));
            doc.add(new Field("Code", Code, getVectorField()));
        }

        if (OwnerUserId != null)
            doc.add(new IntField("OwnerUserId", OwnerUserId, Field.Store.YES));

        if (OwnerDisplayName != null)
            doc.add(new StringField("OwnerDisplayName", OwnerDisplayName, Field.Store.YES));

        if (LastEditorUserId != null)
            doc.add(new IntField("LastEditorUserId", LastEditorUserId, Field.Store.YES));

        if (LastEditorDisplayName != null)
            doc.add(new TextField("LastEditorDisplayName", LastEditorDisplayName, Field.Store.YES));

        if (LastEditDate != null)
            doc.add(new LongField("LastEditDate", LastEditDate.getTime(), Field.Store.YES));

        if (LastActivityDate != null)
            doc.add(new LongField("LastActivityDate", LastActivityDate.getTime(), Field.Store.YES));

        if (ClosedDate != null)
            doc.add(new LongField("ClosedDate", ClosedDate.getTime(), Field.Store.YES));

        if (Title != null)
            doc.add(new TextField("Title", Title, Field.Store.YES));

        if (Tags.size() != 0) {
            for (String tag : Tags)
                doc.add(new StringField("Tags", tag, Field.Store.YES));
        }

        if (Topics.size() != 0) {
            for (Integer topicId : Topics)
                doc.add(new IntField("Topics", topicId, Field.Store.YES));
        }

        if (AnswerCount != null)
            doc.add(new IntField("AnswerCount", AnswerCount, Field.Store.YES));

        if (CommentCount != null) {
            doc.add(new IntField("CommentCount", CommentCount, Field.Store.YES));
            doc.add(new SortedNumericDocValuesField("SortCommentCount", CommentCount));
        }
        if (FavoriteCount != null) {
            doc.add(new IntField("FavoriteCount", FavoriteCount, Field.Store.YES));
            doc.add(new SortedNumericDocValuesField("SortFavoriteCount", FavoriteCount));
        }

        if (CommunityOwnedDate != null)
            doc.add(new LongField("CommunityOwnedDate", CommunityOwnedDate.getTime(), Field.Store.YES));

        return doc;
    }

    @Override
    public String toString() {
        return "Post{\n" + "Id=" + Id +
                ", \nPostTypeId=" + PostTypeId +
                ", \nAcceptedAnswerId=" + AcceptedAnswerId +
                ", \nParentId=" + ParentId +
                ", \nCreationDate=" + CreationDate +
                ", \nScore=" + Score +
                ", \nViewCount=" + ViewCount +
                ", \nBody=" + Body +
                ", \nCode=" + Code +
                ", \nOwnerUserId=" + OwnerUserId +
                ", \nOwnerDisplayName=" + OwnerDisplayName +
                ", \nLastEditorUserId=" + LastEditorUserId +
                ", \nLastEditorDisplayName=" + LastEditorDisplayName +
                ", \nLastEditDate=" + LastEditDate +
                ", \nLastActivityDate=" + LastActivityDate +
                ", \nTitle=" + Title +
                ", \nTags=" + Tags +
                ", \nAnswerCount=" + AnswerCount +
                ", \nCommentCount=" + CommentCount +
                ", \nFavoriteCount=" + FavoriteCount +
                ", \nCommunityOwnedDate=" + CommunityOwnedDate + '}';
    }

}
