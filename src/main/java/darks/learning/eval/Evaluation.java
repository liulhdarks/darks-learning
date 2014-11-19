/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package darks.learning.eval;

import java.util.Set;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;

import darks.learning.common.utils.FreqCount;
import darks.learning.common.utils.FreqMatrix;

/**
 * Evaluate the precision of training model
 * 
 * @author Darks.Liu
 *
 */
public class Evaluation
{
	
	FreqMatrix<Integer> matrix = null;
	
	FreqCount<Integer> TN = null;
	
	FreqCount<Integer> FP = null;
	
	long TP = 0;
	
	long FN = 0;
	
	public void eval(DoubleMatrix reals, DoubleMatrix guesses)
	{
		matrix = new FreqMatrix<Integer>();
		TN = new FreqCount<Integer>();
		FP = new FreqCount<Integer>();
		TP = 0;
		FN = 0;
		int row = Math.min(reals.rows, guesses.rows);
		for (int i = 0; i < row; i++)
		{
			DoubleMatrix real = reals.getRow(i);
			DoubleMatrix guess = guesses.getRow(i);
			int actualIndex = SimpleBlas.iamax(real);
			int guessIndex = SimpleBlas.iamax(guess);
			matrix.add(actualIndex, guessIndex);
			if (actualIndex == guessIndex)
			{
				TP++;
				for (Integer index : matrix.getClasses())
				{
					if (!index.equals(guessIndex))
					{
						TN.addValue(index);
					}
				}
			}
			else
			{
				FN++;
				FP.addValue(guessIndex);
			}
		}
	}
	
	public double precision()
	{
		double p = 0.0;
		for (Integer clazz : matrix.getClasses())
		{
			p += precision(clazz);
		}
		return p / matrix.getClasses().size();
	}
	
	public double precision(int clazz)
	{
		if (TP == 0)
		{
			return 0;
		}
		return (double)TP / ((double)TP + (double)FP.getValue(clazz));
	}
	
	public double accuracy()
	{
		return (TP + TN.totalCount()) / (positive() + negative());
	}
	
	public double recall()
	{
		if (TP == 0)
		{
			return 0;
		}
		return (double)TP / ((double)TP + (double)FN);
	}
	
	public double positive()
	{
		return TP + FP.totalCount();
	}
	
	public double negative()
	{
		return TN.totalCount() + FN;
	}
	
	/**
	 * f1() = fScore(1)
	 * @return
	 */
	public double f1()
	{
		double P = precision();
		double R = recall();
		return 2 * P * R / (P + R);
	}
	
	/**
	 * f-score = (1 + beta^2) * (precision * recall) / (precision * beta^2 + recall)
	 * @param n beta
	 * @return F score
	 */
	public double fScore(int n)
	{
		double P = precision();
		double R = recall();
		return (1 + Math.pow(n, 2)) * P * R / (P * Math.pow(n, 2) + R);
	}
	
	public double TPR()
	{
		return recall();
	}
	
	public double FPR()
	{
		return (double)FP.totalCount() / negative();
	}

	public String status()
	{
		StringBuilder buf = new StringBuilder(128);
		buf.append("Evaluation Result:\n------------------------------------------------\n");
		Set<Integer> clazzes = matrix.getClasses();
		for (Integer actual : clazzes)
		{
			for (Integer guess : clazzes)
			{
				buf.append("Actual class ").append(actual).append(" with Predict class ")
					.append(guess).append(" has ").append(matrix.count(actual, guess)).append(" times\n");
			}
		}
		buf.append("F1 score:").append(f1()).append("\t")
			.append("Accuracy:").append(accuracy())
			.append("\n");
		return buf.toString();
	}
}
