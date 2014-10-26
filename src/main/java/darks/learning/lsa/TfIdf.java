package darks.learning.lsa;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import darks.learning.common.utils.FreqCount;

public class TfIdf
{
    
    private Map<String, Set<String>> wordsMap = new HashMap<String, Set<String>>();
    
    private int totalWordsCount;
    
    private Map<String, FreqCount<String>> sentenceMap = new HashMap<String, FreqCount<String>>();
    
    public void addWord(String sentence, String word)
    {
        Set<String> wordSet = wordsMap.get(word);
        if (wordSet == null)
        {
            wordSet = new HashSet<String>();
            wordsMap.put(word, wordSet);
        }
        wordSet.add(sentence);
        totalWordsCount++;
        FreqCount<String> freq = sentenceMap.get(sentence);
        if (freq == null)
        {
            freq = new FreqCount<String>();
            sentenceMap.put(sentence, freq);
        }
        freq.addValue(word);
    }
    
    public void addWords(String[] words)
    {
        String key = Arrays.toString(words);
        for (String word : words)
        {
            addWord(key, word);
        }
    }
    
    public double getTFIdf(String sentence, String word)
    {
        double tf = getTF(sentence, word);
        double idf = getIDF(word);
        return tf * idf;
    }
    
    public double getTFIdf(FreqCount<String> freq, String word)
    {
        double tf = getTF(freq, word);
        double idf = getIDF(word);
        return tf * idf;
    }
    
    public double getTF(String sentence, String word)
    {
        FreqCount<String> freq = sentenceMap.get(sentence);
        return getTF(freq, word);
    }
    
    public double getTF(FreqCount<String> freq, String word)
    {
        if (freq == null)
        {
            return 0.0001;
        }
        long count = freq.getValue(word);
        long total = freq.totalCount();
        return (double) count / (double) total;
    }
    
    public double getIDF(String word)
    {
        int totalCorpus = sentenceMap.size();
        Set<String> wordSet = wordsMap.get(word);
        int wordInDocCount = wordSet == null ? 0 : wordSet.size();
        return Math.log((double) totalCorpus / (double)(wordInDocCount + 1));
    }

    public int getTotalWordsCount()
    {
        return totalWordsCount;
    }

    public int getTotalSentenceCount()
    {
        return sentenceMap.size();
    }


    public Map<String, Set<String>> getWordsMap()
    {
        return wordsMap;
    }

    public Map<String, FreqCount<String>> getSentenceMap()
    {
        return sentenceMap;
    }
    
    public int getUniqueWordsCount()
    {
        return wordsMap.size();
    }
    
}
