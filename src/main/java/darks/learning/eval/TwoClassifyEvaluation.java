package darks.learning.eval;

public class TwoClassifyEvaluation
{
	
	String posLabel;
	
	String negLabel;
	
	long TP;
	
	long TN;
	
	long FP;
	
	long FN;
	
	public TwoClassifyEvaluation(String posLabel, String negLabel)
	{
		this.posLabel = posLabel;
		this.negLabel = negLabel;
	}
	

	public void eval(String real, String expect)
	{
		if (posLabel.equals(expect)) //P
		{
			if (real.equals(expect))
				TP++;
			else
				FP++;
		}
		else	//N
		{
			if (real.equals(expect))
				FN++;
			else
				TN++;
		}
	}
	
	public double precision()
	{
		return (double) TP / (double)(TP + TN);
	}
	
	public double recall()
	{
		return (double) TP / positive();
	}
	
	public double f1Score()
	{
		double p = precision();
		double r = recall();
		return 2 * p * r / (p + r);
	}
	
	public double accuracy()
	{
		return (double) (TP + FN) / (double) (TP + FP + TN + FN);
	}
	
	public double positive()
	{
		return (double) (TP + FP);
	}
	
	public double negative()
	{
		return (double) (TN + FN);
	}


	public long TP()
	{
		return TP;
	}

	public long TN()
	{
		return TN;
	}


	public long FP()
	{
		return FP;
	}

	public long FN()
	{
		return FN;
	}


	@Override
	public String toString()
	{
		return "TwoClassifyEvaluation [posLabel=" + posLabel + ", negLabel=" + negLabel + ", TP="
				+ TP + ", TN=" + TN + ", FP=" + FP + ", FN=" + FN + "]";
	}
	
}
