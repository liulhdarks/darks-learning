package darks.learning.lsa;

import static darks.learning.common.utils.MatrixHelper.svd;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.Solve;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.MatrixHelper;
import darks.learning.corpus.Corpus;
import darks.learning.exceptions.TrainingException;

public class LatentSemanticAnalysis
{
    
    private static Logger log = LoggerFactory.getLogger(LatentSemanticAnalysis.class);
    
    private Corpus corpus;
    
    private Map<Integer, String> sentenceColumnIndexs = new HashMap<Integer, String>();
    
    private Map<String, Integer> wordsRowIndexs = new HashMap<String, Integer>();
    
    private int K = 300;
    
    DoubleMatrix preMatrix = null;
    
    DoubleMatrix preNorm = null;
    
    DoubleMatrix inverseS = null;
    
    DoubleMatrix Uk = null;
    
    TfIdf tfidf = null;
    
    public void train(Corpus corpus)
    {
        this.corpus = corpus;
        tfidf = corpus.getTfIDF();
        if (tfidf == null)
        {
            throw new TrainingException("LSA training corpus should have TF-IDF type.");
        }
        log.info("Start to train LSA algorithm.");
        long st = System.currentTimeMillis();
        DoubleMatrix trainMatrix = initTrainData();
        log.info("Training LSA...");

        trainMatrix = trainMatrix.getRange(0, 400, 0, 500);
        DoubleMatrix[] USV = svd(trainMatrix);
        
        DoubleMatrix U = USV[0];
        DoubleMatrix S = USV[1];
//        DoubleMatrix V = USV[2];
        Uk = U.getRange(0, U.rows, 0, K); //m*k
        DoubleMatrix Sk = S.getRange(0, K, 0, K); //k*k
//        DoubleMatrix Vk = V.getRange(0, K, 0, V.columns);//k*n

        inverseS = Solve.pinv(Sk);
        preMatrix = trainMatrix.transpose().mmul(Uk).mmul(inverseS); //n*k
        preNorm = MatrixHelper.sqrt(preMatrix.mul(preMatrix).rowSums());
        log.info("Complete to train LSA.preMatrix(" + preMatrix.rows + "," + preMatrix.columns + ") InvS:" 
                            + inverseS.rows + "," + inverseS.columns + ")");
        log.info("LSA training cost " + (System.currentTimeMillis() - st) + "ms");
    }
    
    public String predict(String[] words)
    {
        int column = predictIndex(words);
        return sentenceColumnIndexs.get(column);
    }
    
    public int predictIndex(String[] words)
    {
        DoubleMatrix vector = getSentenceVector(words);
        vector = vector.getRange(0, 400, 0, 1);
        DoubleMatrix vectorMapping = vector.transpose().mmul(Uk).mmul(inverseS);  //1*k
        DoubleMatrix dotVector = vectorMapping.mmul(preMatrix.transpose());
        double norm = vector.norm2();
        DoubleMatrix cosMt = dotVector.div(preNorm.mul(norm));
        return SimpleBlas.iamax(cosMt);
    }
    
    private DoubleMatrix initTrainData()
    {
        int corpusCount = (int)tfidf.getTotalSentenceCount();
        int wordsCount = (int)corpus.getTotalUniqueCount();
        log.debug("Initialize LSA training data " + wordsCount + " * " + corpusCount);
        DoubleMatrix result = new DoubleMatrix(wordsCount, corpusCount);
        int columnIndex = 0;
        int rowIndex = 0;
        for (Entry<String, FreqCount<String>> entryFreq : tfidf.getSentenceMap().entrySet())
        {
            String sentence = entryFreq.getKey();
            FreqCount<String> freq = entryFreq.getValue();
            Iterator<Map.Entry<String, Long>> it = freq.entrySetIterator();
            while (it.hasNext())
            {
                Entry<String, Long> entry = it.next();
                String word = (String)entry.getKey();
                double tfIdfValue = tfidf.getTFIdf(sentence, word);
                Integer wordRowIndex = wordsRowIndexs.get(word);
                if (wordRowIndex == null)
                {
                    wordRowIndex = rowIndex++;
                    wordsRowIndexs.put(word, wordRowIndex);
                }
                result.put(wordRowIndex, columnIndex, tfIdfValue);
            }
            sentenceColumnIndexs.put(columnIndex, entryFreq.getKey());
            columnIndex++;
        }
        return result;
    }
    
    private DoubleMatrix getSentenceVector(String[] sentence)
    {
        int wordsCount = (int)corpus.getTotalUniqueCount();
        DoubleMatrix result = new DoubleMatrix(wordsCount);
        FreqCount<String> freq = new FreqCount<String>();
        for (String s : sentence)
        {
            freq.addValue(s);
        }
        Iterator<Entry<String, Long>> it = freq.entrySetIterator();
        while (it.hasNext())
        {
            Entry<String, Long> entry = it.next();
            String word = (String)entry.getKey();
            Integer index = wordsRowIndexs.get(word);
            if (index == null)
            {
                continue;
            }
            double tfIdf = tfidf.getTFIdf(freq, word);
            result.put(index, tfIdf);
        }
        return result;
    }
}
