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
package darks.learning.classifier.bayes;

/**
 * Native Bayes configuration
 * 
 * @author Darks.Liu
 *
 */
public class NaiveBayesConfig
{

	int modelType = NaiveBayes.BINAMIAL;
	
	int smoothFactor = NaiveBayes.LAPLACE;
	
	boolean logLikelihood = false;
	

	public int getSmoothFactor()
	{
		return smoothFactor;
	}

	public NaiveBayesConfig setSmoothFactor(int smoothFactor)
	{
		this.smoothFactor = smoothFactor;
		return this;
	}

	public boolean isLogLikelihood()
	{
		return logLikelihood;
	}

	public NaiveBayesConfig setLogLikelihood(boolean logLikelihood)
	{
		this.logLikelihood = logLikelihood;
		return this;
	}

	public int getModelType()
	{
		return modelType;
	}

	public NaiveBayesConfig setModelType(int modelType)
	{
		this.modelType = modelType;
		return this;
	}
	
}
