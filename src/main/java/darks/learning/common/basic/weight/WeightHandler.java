package darks.learning.common.basic.weight;

import java.io.Serializable;
import java.util.Collection;

public abstract class WeightHandler implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7356783999373292732L;
	
	protected WeightFilter filter;
	

	public WeightHandler()
	{
	}

	public WeightHandler(WeightFilter filter)
	{
		this.filter = filter;
	}

	public abstract void statSentence(Collection<String> terms);
	
	public abstract double computeWeight(Collection<String> terms, String targetTerm);
	
	
	public static interface WeightFilter extends Serializable
	{
		public boolean filter(String term);
	}
}
