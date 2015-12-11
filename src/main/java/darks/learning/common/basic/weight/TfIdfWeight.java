package darks.learning.common.basic.weight;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import darks.learning.common.utils.FreqCount;

public class TfIdfWeight extends WeightHandler
{
    
    /**
	 * 
	 */
	private static final long serialVersionUID = -8695575116939851071L;
	
	private static final int SMOOTH = 1;

	FreqCount<String> termDocFreq = new FreqCount<String>();
    
    private long totalWordsCount;
    
    private long totalDocCount;
    
    
    
    public TfIdfWeight()
	{
		super();
	}

	public TfIdfWeight(WeightFilter filter)
	{
		super(filter);
	}

	@Override
    public void statSentence(Collection<String> terms)
    {
    	Set<String> uniqueTerms = new HashSet<String>(terms);
    	int validCount = 0;
    	for (String term : uniqueTerms)
    	{
    		if (filter.filter(term))
    			continue;
    		termDocFreq.addValue(term);
    		totalWordsCount++;
    		validCount++;
    	}
    	if (validCount > 0)
    		totalDocCount++;
    }

    @Override
    public double computeWeight(Collection<String> terms, String targetTerm)
    {
        double tf = getTF(terms, targetTerm);
        double idf = getIDF(targetTerm);
        return tf * idf;
    }
    
    public double getTF(Collection<String> terms, String word)
    {
    	int count = 0;
    	for (String term : terms)
    	{
    		if (word.equals(term))
    			count++;
    	}
        return (double) count / (double) terms.size();
    }
    
    public double getTF(FreqCount<String> freq, String word)
    {
        if (freq == null)
        {
            return 0.00001;
        }
        long count = freq.getValue(word);
        long total = freq.totalCount();
        return (double) count / (double) total;
    }
    
    public double getIDF(String word)
    {
        long wordRefCount = termDocFreq.getValue(word);
        return Math.log((double) totalDocCount / (double)(wordRefCount + SMOOTH));
    }

	public FreqCount<String> getTermDocFreq()
	{
		return termDocFreq;
	}

	public void setTermDocFreq(FreqCount<String> termDocFreq)
	{
		this.termDocFreq = termDocFreq;
	}

	public long getTotalWordsCount()
	{
		return totalWordsCount;
	}

	public void setTotalWordsCount(long totalWordsCount)
	{
		this.totalWordsCount = totalWordsCount;
	}

	public long getTotalDocCount()
	{
		return totalDocCount;
	}

	public void setTotalDocCount(long totalDocCount)
	{
		this.totalDocCount = totalDocCount;
	}
    
}
