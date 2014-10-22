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
package darks.learning;

import darks.learning.common.rand.JdkRandomFunction;
import darks.learning.common.rand.RandomFunction;
import darks.learning.lossfunc.LossFunction;

/**
 * Learning algorithm basic configuration
 * 
 * @author Darks.Liu
 *
 */
public abstract class LearningConfig
{

	
	public RandomFunction randomFunction = new JdkRandomFunction();
	
	public boolean normalized = false;
	
	public LossFunction lossFunction = null;
	
	public int lossType;
	
	public boolean useRegularization = true;
	
	public double L2 = 0.1;
}
