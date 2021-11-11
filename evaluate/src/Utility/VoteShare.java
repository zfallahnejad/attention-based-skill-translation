package Utility;

import org.jsoup.Jsoup;
import org.jsoup.select.Elements;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class VoteShare {
    public void compute_voteshare(String tag, String XMLInput) {
        try {
            File dir = new File(Constants.Voteshare_Directory + tag + "/");
            if (!dir.exists())
                dir.mkdirs();

            BufferedReader reader = new BufferedReader(new FileReader(XMLInput));
            String line;
            HashMap<Integer, ArrayList<Integer>> question_answers = new HashMap<>();
            HashMap<Integer, Integer> answer_score = new HashMap<>();
            HashMap<Integer, Integer> answer_parent = new HashMap<>();
            ArrayList<Integer> question_list = new ArrayList<>();
            ArrayList<Integer> answer_list = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                if (line.trim().startsWith("<row")) {
                    Elements row = Jsoup.parse(line).getElementsByTag("row");
                    Integer Id = Integer.parseInt(row.attr("Id"));
                    Integer PostTypeId = Integer.parseInt(row.attr("PostTypeId"));
                    if (PostTypeId == 2) {
                        // answers
                        answer_list.add(Id);
                        Integer ParentId = Integer.parseInt(row.attr("ParentId"));
                        Integer Score = Integer.parseInt(row.attr("Score"));

                        answer_score.put(Id, Score);
                        answer_parent.put(Id, ParentId);
                        if (question_answers.containsKey(ParentId)) {
                            question_answers.get(ParentId).add(Id);
                        } else {
                            ArrayList<Integer> ans = new ArrayList<>();
                            ans.add(Id);
                            question_answers.put(ParentId, ans);
                            question_list.add(ParentId);
                        }
                    }
                }
            }
            reader.close();

            PrintWriter out;
            // out = new PrintWriter(Constants.Voteshare_Directory + tag + "/" + tag + "_question_answers_v1.csv");
            // for (Integer qid : question_list) {
            //     ArrayList<Integer> ans = question_answers.get(qid);
            //     String answers = "";
            //     for (Integer aid : ans) {
            //         answers += "," + aid;
            //     }
            //     out.println(qid + answers);
            // }
            // out.close();

            // out = new PrintWriter(Constants.Voteshare_Directory + tag + "/" + tag + "_answer_scores_v1.csv");
            // for (Integer aid : answer_list) {
            //     out.println(aid + "," + answer_score.get(aid));
            // }
            // out.close();

            out = new PrintWriter(Constants.Voteshare_Directory + tag + "/" + tag + "_vote_share_v1.csv");
            HashMap<Integer, Integer> question_sum_scores = new HashMap<>();
            for (Integer qid : question_list) {
                Integer sum_score = 0;
                for (Integer aid : question_answers.get(qid)) {
                    sum_score += answer_score.get(aid);
                }
                question_sum_scores.put(qid, sum_score);
            }
            for (Integer aid : answer_list) {
                double voteshare = 0.0;
                if (question_sum_scores.get(answer_parent.get(aid)) != 0) {
                    voteshare = (answer_score.get(aid) * 1.0 / question_sum_scores.get(answer_parent.get(aid)));
                }
                if (String.valueOf(voteshare).length() <= 4) {
                    out.println(aid + "," + voteshare);
                    System.out.println(aid + "," + voteshare);
                } else {
                    out.println(aid + "," + String.format("%.4f", voteshare));
                    System.out.println(aid + "," + String.format("%.4f", voteshare));
                }
            }
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
