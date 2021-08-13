package Index;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

public class LuceneIndex {
    private Analyzer analyzer;
    private Directory fsDir;
    private Directory ramDir;
    private IndexWriter ramWriter, fileWriter;
    private IndexWriterConfig config;
    private LMJelinekMercerSimilarity sim;
    private int maxDocInMemory = 100000;
    private int countInMemoryDoc = 0;

    public void setUp(String indexPath) {
        try {
            analyzer = new StandardAnalyzer();
            ramDir = new RAMDirectory();
            fsDir = FSDirectory.open(Paths.get(indexPath));
            IndexWriterConfig config1 = new IndexWriterConfig(analyzer);
            fileWriter = new IndexWriter(fsDir, config1);
            config = new IndexWriterConfig(analyzer);
            ramWriter = new IndexWriter(ramDir, config);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void index(String xmlFile) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(xmlFile));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().startsWith("<row")) {
                    Post post = new Post(line);
                    addToIndex(post.getLuceneDocument());
                }
            }
            reader.close();
            closeIndex();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void addToIndex(Document document) throws IOException {
        if (countInMemoryDoc < maxDocInMemory) {
            // Add to Ram Memory and count up
            ramWriter.addDocument(document);
            countInMemoryDoc++;
        } else {
            System.out.println("Making index for " + countInMemoryDoc + " Documents.");
            // Merge Ram Memory and create a new ram memory
            ramWriter.addDocument(document);
            ramWriter.close();
            fileWriter.addIndexes(ramDir);
            ramDir.close();
            ramDir = new RAMDirectory();
            ramWriter = new IndexWriter(ramDir, new IndexWriterConfig(analyzer));
            countInMemoryDoc = 0;
        }
    }

    private void closeIndex() throws IOException {
        ramWriter.close();
        fileWriter.addIndexes(ramDir);
        ramDir.close();
        countInMemoryDoc = 0;
        fileWriter.close();
        fsDir.close();
    }
}
