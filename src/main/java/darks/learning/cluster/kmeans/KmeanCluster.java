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
package darks.learning.cluster.kmeans;

import java.util.List;

import darks.learning.cluster.ClusterExecutor;
import darks.learning.distance.Distance;

public class KmeanCluster<T> extends ClusterExecutor<T>
{

	
	public KmeanCluster()
	{
		super();
	}

	public KmeanCluster(Distance<T> distance)
	{
		super(distance);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void cluster(List<T> records)
	{
		// TODO Auto-generated method stub

	}

}
