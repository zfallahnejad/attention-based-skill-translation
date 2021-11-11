## 1. extract `rar` file inside Voteshare/java and Voteshare/php directories.
* Voteshare1 = vote / sum of vote scores in the thread 
* Voteshare2 produce same value as nobari = vote / sum of positive vote scores in the thread  
* Voteshare3 apply min max normalization to votes
* Voteshare4 shift vote scores to make sure the smallest (modified) vote-score is zero then calculate voteshare
* Voteshare5 shift vote scores to make sure the smallest (modified) vote-score is one then calculate voteshare
## 2. run `Main.java` which create lucene index for java and php collection and find experts using astm skill translations
