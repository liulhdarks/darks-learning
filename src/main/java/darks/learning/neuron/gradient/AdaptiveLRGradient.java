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
package darks.learning.neuron.gradient;

import static darks.learning.common.utils.MatrixHelper.pow;
import static darks.learning.common.utils.MatrixHelper.sqrt;
import static darks.learning.common.utils.MatrixHelper.abs;
import org.jblas.DoubleMatrix;

/**
 * Adaptive learn rate gradient
 * 
 * @author Darks.Liu
 *
 */
public class AdaptiveLRGradient
{

	double stepSize = 1e-3;
	
	DoubleMatrix historicalGrads;
	
	DoubleMatrix currentGrad;
	
	double smooth = 1e-6;
	
	DoubleMatrix learnRates;
	
	public AdaptiveLRGradient()
	{
		this(1e-2);
	}
	
	public AdaptiveLRGradient(double stepSize)
	{
		this.stepSize = stepSize;
	}
	
	public AdaptiveLRGradient(int rows, int columns)
	{
		this(rows, columns, 1e-2);
	}
	
	public AdaptiveLRGradient(int rows, int columns, double stepSize)
	{
		this.stepSize = stepSize;
		historicalGrads = new DoubleMatrix(rows, columns);
	}
	
	
	public DoubleMatrix getLearnRates(DoubleMatrix gradient)
	{
		currentGrad = gradient.dup();
		if (historicalGrads == null || historicalGrads.length != gradient.length)
		{
			historicalGrads = new DoubleMatrix(currentGrad.rows, currentGrad.columns);
		}
		historicalGrads.addi(pow(currentGrad, 2));
		DoubleMatrix sqrtGrad = sqrt(historicalGrads).add(smooth);
		learnRates = abs(currentGrad).div(sqrtGrad).mul(stepSize);
		return learnRates;
	}
}
