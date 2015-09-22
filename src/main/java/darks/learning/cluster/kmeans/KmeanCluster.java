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

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig.OptimizeType;
import darks.learning.cluster.ClusterExecutor;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

public class KmeanCluster<T> extends ClusterExecutor<T>
{

	private static final Logger log = LoggerFactory.getLogger(KmeanCluster.class);
	
	List<Cluster<T>> clusters;
	
	List<ClusterPoint<T>> points;
	
	ClusterScanner<T> clusterScanner = null;

	public KmeanCluster(Distance<T> distance)
	{
		super(distance);
			
	}
	
	private void initialize(List<T> records)
	{
		if (config.getOptimizeType() == OptimizeType.TRIANGLE_INEQUALITY)
			clusterScanner = new TriangleClusterScanner<T>(distance);
		else
			clusterScanner = new FullClusterScanner<T>(distance);
		points = new ArrayList<ClusterPoint<T>>();
		clusters = new ArrayList<Cluster<T>>();
		for (T t : records)
		{
			points.add(new ClusterPoint<T>(t));
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void cluster(List<T> records)
	{
		initialize(records);
		int clusterCount = Math.min(config.getClusterCount(), records.size());
		for (int i = 0; i < clusterCount; i++)
		{
			ClusterPoint<T> center = points.get(i);
			center.setCenterPoint(true);
			clusters.add(new Cluster<T>(center));
		}
		if (clusterCount >= points.size())
			return;
		double lastLastCohesiveValue = 0;
		double lastCohesiveValue = 0;
		for (int epoch = 1; epoch <= config.getIterateCount(); epoch++)
		{
			for (Cluster<T> cluster : clusters)
			{
				cluster.getPoints().clear();
			}
			long costTime = clusterScanner.scanCluster(clusters, points);
			double cohesiveValue = optimizeCenter();
			log.info("Iteration " + epoch + " cohesive value " + cohesiveValue + " cost:" + costTime + "ms");
			if (cohesiveValue == lastCohesiveValue || cohesiveValue == lastLastCohesiveValue)
				break;
			lastLastCohesiveValue = lastCohesiveValue;
			lastCohesiveValue = cohesiveValue;
		}
		
	}
	
	private double optimizeCenter()
	{
		double cohesiveValue = 0; 
		//adjust center points
		for (Cluster<T> cluster : clusters)
		{
			cluster.setLowCluster(null);
			cluster.getClusterMap().clear();
			double finalSum = 0;
			double minSum = 0;
			ClusterPoint<T> minPoint = null;
			for (ClusterPoint<T> point : cluster.getPoints())
			{
				double sum = 0;
				for (ClusterPoint<T> target : cluster.getPoints())
				{
					if (target.equals(point))
						continue;
					sum += distance.distance(target.getObject(), point.getObject());
				}
				if (minPoint == null || sum < minSum)
				{
					minSum = sum;
					minPoint = point;
				}
			}
			finalSum = minSum;
			if (minPoint != null)
			{
				double centerSum = 0;
				for (ClusterPoint<T> target : cluster.getPoints())
				{
					centerSum += distance.distance(target.getObject(), cluster.getCenter().getObject());
				}
				if (centerSum > minSum)
				{
					cluster.getCenter().setCenterPoint(false);
					cluster.getPoints().add(cluster.getCenter());
					cluster.getPoints().remove(minPoint);
					minPoint.setCenterPoint(true);
					cluster.setCenter(minPoint);
					finalSum = minSum;
				}
				else
				{
					finalSum = centerSum;
				}
			}
			cohesiveValue += finalSum;
		}
		return cohesiveValue;
	}

	@Override
	public Collection<Cluster<T>> getClusters()
	{
		return clusters;
	}

}
