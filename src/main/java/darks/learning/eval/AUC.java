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

import darks.learning.eval.ROC.PlotLine;
import darks.learning.exceptions.EvaluationException;

/**
 * Area Under ROC Curve
 * 
 * @author Darks.Liu
 *
 */
public class AUC
{
	
	ROC roc;
	
	int lineId;

	public AUC()
	{
		
	}

	public AUC(ROC roc, int lineId)
	{
		this.roc = roc;
		this.lineId = lineId;
	}

	/**
	 * Compute ROC'S AUC value
	 */
	public double compute()
	{
		return compute(roc, lineId);
	}
	
	/**
	 * Compute ROC'S AUC value
	 * 
	 * @param roc ROC line
	 * @param lineId Line id
	 * @return AUC value
	 */
	public double compute(ROC roc, int lineId)
	{
		PlotLine line = roc.plotsMap.get(lineId);
		if (line == null)
		{
			throw new EvaluationException("Cannot find ROC line " + lineId);
		}
		double sum = 0;
		ROCPoint lastPlot = line.zeroPlot;
		while (lastPlot != null)
		{
			ROCPoint curPoint = lastPlot.next;
			if (curPoint != null)
			{
				sum += (curPoint.tpr + lastPlot.tpr) * Math.abs(curPoint.fpr - lastPlot.fpr) / 2.;
				if (curPoint.next == null)
				{
					sum += (1 + curPoint.tpr) * Math.abs(1 - curPoint.fpr) / 2.;
				}
			}
			lastPlot = curPoint;
		}
		return sum;
	}
	
}
