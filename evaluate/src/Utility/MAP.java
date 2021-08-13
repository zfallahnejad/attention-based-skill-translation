package Utility;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by Zohreh on 6/30/2017.
 */
public class MAP {

    public String computeMAP(String tag, String GoldenSetDirectory, String DirName, String fileName, String format) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(GoldenSetDirectory + tag + ".csv"));
        ArrayList<String> goldenMeasure = new ArrayList<>();
        String line = "";
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            goldenMeasure.add(parts[0]);
        }
        //System.out.println(goldenMeasure);
        reader.close();

        BufferedReader reader2 = new BufferedReader(new FileReader(DirName + "/" + fileName + "_" + tag + "." + format));
        String line2 = "";
        ArrayList<result> list = new ArrayList<result>();

        while ((line2 = reader2.readLine()) != null) {
            String[] parts = line2.split(",");
            result s = new result(parts[0], Double.parseDouble(parts[parts.length-1]), goldenMeasure.contains(parts[0]));
            list.add(s);
        }
        reader2.close();
        Collections.sort(list);
        int relevant = 0;
        double sum = 0;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i).isExpert) relevant++;
            double pati = relevant / (i + 1.0);
            if (list.get(i).isExpert) sum += pati;
        }
        System.out.println(tag + "," + relevant + "," + goldenMeasure.size() + "," + sum / goldenMeasure.size());
        return tag + "," + relevant + "," + goldenMeasure.size() + "," + sum / goldenMeasure.size();
    }

    public String computePat(String tag, String GoldenSetDirectory, String DirName, String fileName, int n, String format) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(GoldenSetDirectory + tag + ".csv"));
        ArrayList<String> goldenMeasure = new ArrayList<>();
        String line = "";
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            goldenMeasure.add(parts[0]);
        }
        // System.out.println(goldenMeasure);
        reader.close();

        BufferedReader reader2 = new BufferedReader(new FileReader(DirName + "/" + fileName + "_" + tag + "." + format));
        String line2 = "";
        ArrayList<result> list = new ArrayList<result>();

        while ((line2 = reader2.readLine()) != null) {
            String[] parts = line2.split(",");
            result s = new result(parts[0], Double.parseDouble(parts[parts.length-1]), goldenMeasure.contains(parts[0]));
            list.add(s);
        }
        reader2.close();
        Collections.sort(list);
        int relevant = 0;
        int len = list.size() < n ? list.size() : n;
        for (int i = 0; i < len; i++)
            if (list.get(i).isExpert) relevant++;
        double pati = relevant * 1.0 / n;
        System.out.println(tag + "," + relevant + "," + n + "," + pati);
        return tag + "," + relevant + "," + n + "," + pati;
    }

    class result implements Comparable<result> {
        String eid;
        Double score;
        boolean isExpert;

        public result(String eid, Double score, boolean contains) {
            this.eid = eid;
            this.isExpert = contains;
            if (isExpert)
                this.score = score + 0.0001;
            else
                this.score = score;

        }

        @Override
        public int compareTo(result o) {
            return -1 * Double.compare(score, o.score);
        }
    }
}
