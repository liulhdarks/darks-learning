package darks.learning.topic.lsa;

import static darks.learning.common.utils.MatrixHelper.svd;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.jblas.Solve;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.basic.TfIdf;
import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.IOUtils;
import darks.learning.common.utils.MatrixHelper;
import darks.learning.corpus.Corpus;
import darks.learning.exceptions.TrainingException;

/**
 * Latent Semantic Analysis
 * @author Darks.Liu
 *
 */
public class LatentSemanticAnalysis
{
    
    private static Logger log = LoggerFactory.getLogger(LatentSemanticAnalysis.class);
    
    private Map<Integer, String> sentenceColumnIndexs = new HashMap<Integer, String>();
    
    private Map<String, Integer> wordsRowIndexs = new HashMap<String, Integer>();
    
    private int K = 300;
    
    DoubleMatrix preMatrix = null;
    
    DoubleMatrix preNorm = null;
    
    DoubleMatrix inverseS = null;
    
    DoubleMatrix Uk = null;
    
    TfIdf tfidf = null;
    
    public LatentSemanticAnalysis()
    {
    	
    }
    
    /**
     * Construction
     * @param targetDimension Target reduce dimension.Default 300
     */
    public LatentSemanticAnalysis(int targetDimension)
    {
    	K = targetDimension > 0 ? targetDimension : K;
    }
    
    /**
     * Train model by corpus
     * @param corpus {@linkplain darks.learning.corpus.Corpus Corpus}
     */
    public void train(Corpus corpus)
    {
    	tfidf = corpus.getTfIDF();
        DoubleMatrix trainMatrix = initTrainData();
        train(tfidf, trainMatrix);
    }
    
    public void train(TfIdf tfidf, DoubleMatrix trainMatrix)
    {
        if (tfidf == null)
        {
            throw new TrainingException("LSA training corpus should have TF-IDF type.");
        }
        log.info("Start to train LSA algorithm.");
        long st = System.currentTimeMillis();
        log.info("Training LSA...");
//        trainMatrix = trainMatrix.getRange(0, 400, 0, 500);
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


    /**
     * Predict target sentence content by words' array
     * @param words Words' array
     * @return Target sentence content
     */
    public String predict(String[] words)
    {
        int column = predictIndex(words);
        return sentenceColumnIndexs.get(column);
    }

    /**
     * Predict target index by words' array
     * @param words Words' array
     * @return Target index
     */
    public int predictIndex(String[] words)
    {
        DoubleMatrix vector = getSentenceVector(words);
        return predictIndex(vector);
    }
    
    /**
     * Predict target index by vector
     * @param vector Feature vector
     * @return Target index
     */
    public int predictIndex(DoubleMatrix vector)
    {
//        vector = vector.getRange(0, 400, 0, 1);
        DoubleMatrix vectorMapping = vector.transpose().mmul(Uk).mmul(inverseS);  //1*k
        DoubleMatrix dotVector = vectorMapping.mmul(preMatrix.transpose());
        double norm = vector.norm2();
        DoubleMatrix cosMt = dotVector.div(preNorm.mul(norm));
        return SimpleBlas.iamax(cosMt);
    }
    
    /**
     * Save LSA model
     * @param file Target model file
     */
    public void saveModel(File file)
    {
    	LsaModel model = new LsaModel(sentenceColumnIndexs, wordsRowIndexs, K, 
    				preMatrix, preNorm, inverseS, Uk, tfidf);
    	ObjectOutputStream oos = null;
    	try
		{
        	oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
        	oos.writeObject(model);
        	oos.flush();
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
    	finally
    	{
    		IOUtils.closeStream(oos);
    	}
    	
    }
    
    /**
     * Load LSA model from file
     * @param file Model file 
     */
    public void loadModel(File file)
    {
    	ObjectInputStream ois = null;
    	try
		{
    		ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(file)));
    		LsaModel model = (LsaModel) ois.readObject();
    		sentenceColumnIndexs = model.getSentenceColumnIndexs();
    		wordsRowIndexs = model.getWordsRowIndexs();
    		K = model.getK();
    		preMatrix = model.getPreMatrix();
    		preNorm = model.getPreNorm();
    		inverseS = model.getInverseS();
    		Uk = model.getUk();
    		tfidf = model.getTfidf();
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
		}
    	finally
    	{
    		IOUtils.closeStream(ois);
    	}
    }
    
    private DoubleMatrix initTrainData()
    {
        int corpusCount = (int)tfidf.getTotalSentenceCount();
        int wordsCount = (int)tfidf.getUniqueWordsCount();
        log.debug("Initialize LSA training data " + corpusCount + " * " + wordsCount);
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
        int wordsCount = (int)tfidf.getUniqueWordsCount();
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
    
	public Map<Integer, String> getSentenceColumnIndexs()
	{
		return sentenceColumnIndexs;
	}

	public Map<String, Integer> getWordsRowIndexs()
	{
		return wordsRowIndexs;
	}
    
    
}
