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
package darks.learning.cluster.hierarchy;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterExecutor;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

public class DynamicalCluster<T> extends ClusterExecutor<T>
{
	
	static Logger log = LoggerFactory.getLogger(DynamicalCluster.class);
	
	Set<Cluster<T>> clusters = new HashSet<Cluster<T>>();
	
	Set<ClusterPoint<T>> pointsCache = new HashSet<ClusterPoint<T>>();
	
	public DynamicalCluster()
	{
		
	}
	
	public DynamicalCluster(Distance<T> distance)
	{
		super(distance);
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void clusterIncrement(T record)
	{
		addPoint(new ClusterPoint<T>(record));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void cluster(List<T> records)
	{
		for (T point : records)
		{
			clusterIncrement(point);
		}
		optimizeCluster();
	}
	
	public void optimizeCluster()
	{
		int lastCount = clusters.size();
		for (int i = 0; i < config.getIterateCount(); i++)
		{
			if (log.isDebugEnabled())
			{
				log.debug("Repeat optimize cluster iteration:" + (i + 1) + " points:" + getRecordCount());
			}
			pointsCache.clear();
			List<Cluster<T>> clustersTemp = new LinkedList<Cluster<T>>(clusters);
			clusters.clear();
			int index = 0;
			for (Cluster<T> cluster : clustersTemp)
			{
				adjustCenter(cluster);
				clusters.add(cluster);
				pointsCache.addAll(cluster.getPoints());
				cluster.getPoints().clear();
				if (log.isDebugEnabled() && ++index % 100 == 0)
				{
					log.debug("Adjust cluster center:" + ((float)index / (float)clustersTemp.size() * 100) + "%");
				}
			}
			clustersTemp.clear();
//			for (Cluster<T> cluster : clusters)
//			{
//				pointsCache.addAll(cluster.getPoints());
//				cluster.getPoints().clear();
//			}
			mergeCluster();
			if (log.isDebugEnabled())
			{
				log.debug("Repeat add point to clusters " + clusters.size());
			}
			index = 0;
			for (ClusterPoint<T> point : pointsCache)
			{
				addPoint(point);
				if (log.isDebugEnabled() && ++index % 1000 == 0)
				{
					log.debug("Repeat add point percent:" + ((float)index / (float)pointsCache.size() * 100) + "%");
				}
			}
			if (lastCount == clusters.size())
			{
				break;
			}
			lastCount = clusters.size();
		}
	}
	
	private int getRecordCount()
	{
		int count = 0;
		for (Cluster<T> cluster : clusters)
		{
			count++;
			count += cluster.getPoints().size();
		}
		return count;
	}
	
	private void mergeCluster()
	{
		if (log.isDebugEnabled())
		{
			log.debug("Before merge cluster:" + clusters.size());
		}
		Set<Cluster<T>> clustersTemp = new HashSet<Cluster<T>>();
		for (Cluster<T> cluster : clusters)
		{
			ClusterPoint<T> point = cluster.getCenter();
			Cluster<T> maxCluster = null;
			double matchRate = 0;
			for (Cluster<T> temp : clustersTemp)
			{
				double rate = distancePoints(point, temp.getCenter());
				if (maxCluster == null || rate > matchRate)
				{
					matchRate = rate;
					maxCluster = temp;
				}
			}
			if (maxCluster != null && matchRate >= config.getMergeSimilar())
			{
				point.setSimilar(matchRate);
				point.setCenterPoint(false);
				maxCluster.getPoints().add(point);
			}
			else
			{
				clustersTemp.add(cluster);
			}
		}
		clusters = clustersTemp;
		if (log.isDebugEnabled())
		{
			log.debug("After merge cluster:" + clusters.size());
		}
	}
	
	private void addPoint(ClusterPoint<T> point)
	{
		Cluster<T> maxCluster = null;
		double matchRate = 0;
		for (Cluster<T> cluster : clusters)
		{
			double rate = distancePoints(point, cluster.getCenter());
			if (maxCluster == null || rate > matchRate)
			{
				matchRate = rate;
				maxCluster = cluster;
			}
		}
		if (maxCluster != null && matchRate >= config.getMinSimilar())
		{
			point.setCenterPoint(false);
			point.setSimilar(matchRate);
			maxCluster.getPoints().add(point);
		}
		else
		{
			Cluster<T> newCluster = new Cluster<T>();
			newCluster.setCenter(point);
			point.setCenterPoint(true);
			clusters.add(newCluster);
		}
	}
	
	private Cluster<T> adjustCenter(Cluster<T> cluster)
	{
		List<ClusterPoint<T>> points = new LinkedList<ClusterPoint<T>>(cluster.getPoints());
		points.add(cluster.getCenter());
		ClusterPoint<T> maxPoint = null;
		double maxRate = 0;
		for (ClusterPoint<T> point : points)
		{
			double sum = 0;
			for (ClusterPoint<T> target : points)
			{
				sum += distancePoints(point, target);
			}
			if (maxPoint == null || sum > maxRate)
			{
				maxPoint = point;
				maxRate = sum;
			}
		}
		if (!maxPoint.equals(cluster.getCenter()))
		{
			cluster.getCenter().setCenterPoint(false);
			cluster.getPoints().add(cluster.getCenter());
			cluster.getPoints().remove(maxPoint);
			cluster.setCenter(maxPoint);
			maxPoint.setCenterPoint(true);
		}
		return cluster;
	}
	
	private double distancePoints(ClusterPoint<T> point, ClusterPoint<T> target)
	{
		return distance.distance(point.getObject(), target.getObject()); 
	}

	@Override
	public Collection<Cluster<T>> getClusters()
	{
		return clusters;
	}
	
}
