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
package darks.learning.classifier.regression;

import org.jblas.DoubleMatrix;

import darks.learning.SupervisedLearning;
import darks.learning.model.ModelSet;
import darks.learning.neuron.gradient.AdaptiveLRGradient;

/**
 * Regression model
 * @author Darks.Liu
 *
 */
public abstract class Regression implements SupervisedLearning
{
	
	public RegressionConfig config = new RegressionConfig();
	
	DoubleMatrix weights = null;
	
	DoubleMatrix bias = null;
	
	double learnRate;

	AdaptiveLRGradient adaGrad;
	
	/**
	 * Train model set
	 * 
	 * @param modelSet Model set
	 */
	public void trainBatch(ModelSet modelSet)
	{
		trainBatch(modelSet.getInput(), modelSet.getOutput());
	}
	
	protected void gradientDescent(DoubleMatrix input, DoubleMatrix output)
	{
		DoubleMatrix f = config.activateFunction.activate(input.mmul(weights).addRowVector(bias));
		DoubleMatrix error = output.sub(f);
		DoubleMatrix theta = null;
		if (config.useAdaGrad)
		{
			if (adaGrad == null)
			{
				adaGrad = new AdaptiveLRGradient(error.rows, error.columns);
			}
			theta = error.mul(adaGrad.getLearnRates(error));
		}
		else
		{
			theta = error.mul(learnRate);
		}
		DoubleMatrix delta = input.transpose().mmul(theta);
		theta = theta.columnSums();
		if (config.normalized)
		{
			delta.divi(input.rows);
			theta.divi(input.rows);
		}
		weights.addi(delta);
		bias.addi(theta);
	}
	
}
