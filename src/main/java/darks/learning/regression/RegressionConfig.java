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
package darks.learning.regression;

import darks.learning.common.rand.JavaRandomFunction;
import darks.learning.common.rand.RandomFunction;

/**
 * Regression model configuration
 * 
 * @author Darks.Liu
 *
 */
public class RegressionConfig
{

	public double learnRate;
	
	public int maxIteratorCount = 500;
	
	public double minError = 0.00001;
	
	public boolean randomGradient = false;
	
	public RandomFunction randomFunction = new JavaRandomFunction();
	
	public RegressionConfig()
	{
		
	}

	/**
	 * Set learn rate
	 * @param learnRate Learn rate
	 * @return this
	 */
	public RegressionConfig setLearnRate(double learnRate)
	{
		this.learnRate = learnRate;
		return this;
	}

	public RegressionConfig setMaxIteratorCount(int maxIteratorCount)
	{
		this.maxIteratorCount = maxIteratorCount;
		return this;
	}

	public RegressionConfig setMinError(double minError)
	{
		this.minError = minError;
		return this;
	}

	public RegressionConfig setRandomGradient(boolean randomGradient)
	{
		this.randomGradient = randomGradient;
		return this;
	}

	public void setRandomFunction(RandomFunction randomFunction)
	{
		this.randomFunction = randomFunction;
	}
	
}
