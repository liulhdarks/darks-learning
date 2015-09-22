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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Cluster<T>
{

	ClusterPoint<T> center;
	
	Set<ClusterPoint<T>> points = new HashSet<ClusterPoint<T>>();
	
	Map<Cluster<T>, Double> clusterMap = new HashMap<Cluster<T>, Double>();
	
	Cluster<T> lowCluster = null;
	
	double lowDistance = 0;
	
	public Cluster()
	{
	}
	
	public Cluster(T t)
	{
		center = new ClusterPoint<T>(t, 1.D);
	}
	
	public Cluster(ClusterPoint<T> center)
	{
		this.center = center;
	}

	public ClusterPoint<T> getCenter()
	{
		return center;
	}

	public void setCenter(ClusterPoint<T> center)
	{
		this.center = center;
	}

	public Set<ClusterPoint<T>> getPoints()
	{
		return points;
	}

	public void setPoints(Set<ClusterPoint<T>> points)
	{
		this.points = points;
	}
	
	public Map<Cluster<T>, Double> getClusterMap()
	{
		return clusterMap;
	}
	

	public Cluster<T> getLowCluster()
	{
		return lowCluster;
	}

	public void setLowCluster(Cluster<T> lowCluster)
	{
		this.lowCluster = lowCluster;
	}

	public double getLowDistance()
	{
		return lowDistance;
	}

	public void setLowDistance(double lowDistance)
	{
		this.lowDistance = lowDistance;
	}

	@Override
	public int hashCode()
	{
		final int prime = 31;
		int result = 1;
		result = prime * result + ((center == null) ? 0 : center.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		@SuppressWarnings("unchecked")
		Cluster<T> other = (Cluster<T>) obj;
		if (center == null)
		{
			if (other.center != null)
				return false;
		}
		else if (!center.equals(other.center))
			return false;
		return true;
	}

	@Override
	public String toString()
	{
		return "Cluster [center=" + center + ", lowCluster=" + lowCluster + ", lowDistance="
				+ lowDistance + "]";
	}
	
	
	
}
