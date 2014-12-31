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
package darks.learning.cluster;

public class ClusterConfig
{

	double minSimilar = 0.8;
	
	double mergeSimilar = 0.75;
	
	int iterateCount = 5;

	public double getMinSimilar()
	{
		return minSimilar;
	}

	public ClusterConfig setMinSimilar(double minSimilar)
	{
		this.minSimilar = minSimilar;
		return this;
	}

	public double getMergeSimilar()
	{
		return mergeSimilar;
	}

	public ClusterConfig setMergeSimilar(double mergeSimilar)
	{
		this.mergeSimilar = mergeSimilar;
		return this;
	}

	public int getIterateCount()
	{
		return iterateCount;
	}

	public ClusterConfig setIterateCount(int iterateCount)
	{
		this.iterateCount = iterateCount;
		return this;
	}
	
	
}
