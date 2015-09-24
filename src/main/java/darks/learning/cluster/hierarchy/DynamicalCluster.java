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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig.OptimizeType;
import darks.learning.cluster.ClusterExecutor;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

public class DynamicalCluster<T> extends ClusterExecutor<T>
{
	
	static Logger log = LoggerFactory.getLogger(DynamicalCluster.class);
	
	List<Cluster<T>> clusters = new ArrayList<Cluster<T>>();
	
	Set<ClusterPoint<T>> pointsCache = new HashSet<ClusterPoint<T>>();
	
	ClusterScanner<T> clusterScanner = null;
	
	public DynamicalCluster(Distance<T> distance)
	{
		super(distance);
	}
	
	private void initialize()
	{
		if (clusterScanner == null)
		{
			if (config.getOptimizeType() == OptimizeType.TRIANGLE_INEQUALITY)
				clusterScanner = new TriangleClusterScanner<T>(distance, config);
			else
				clusterScanner = new FullClusterScanner<T>(distance, config);
		}
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void clusterIncrement(T record)
	{
		initialize();
		clusterScanner.scanCluster(clusters, new ClusterPoint<T>(record));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void cluster(List<T> records)
	{
		initialize();
		log.info("Start to cluster " + records.size() + " records.");
		long costTime = 0;
		for (T point : records)
		{
			costTime += clusterScanner.scanCluster(clusters, new ClusterPoint<T>(point));
		}
		log.info("cluster first cost time " + costTime + "ms");
//		costTime += optimizeCluster();
		log.info("Complete to cluster records.clusters:" + clusters.size() + " costTime:" + costTime + "ms");
	}
	
	public long optimizeCluster()
	{
		long rt = System.currentTimeMillis();
		double cohesiveValue = 0; 
		double lastCohesiveValue = 0;
		for (int i = 0; i < config.getIterateCount(); i++)
		{
			if (log.isDebugEnabled())
			{
				log.debug("Repeat optimize cluster iteration:" + (i + 1) + " last_cohesive:" + cohesiveValue);
			}
			pointsCache.clear();
			List<Cluster<T>> clustersTemp = new LinkedList<Cluster<T>>(clusters);
			clusters.clear();
			long st = System.currentTimeMillis();
			int index = 0;
			cohesiveValue = 0; 
			for (Cluster<T> cluster : clustersTemp)
			{
				cohesiveValue += adjustCenter(cluster);
				clusters.add(cluster);
				pointsCache.addAll(cluster.getPoints());
				cluster.getPoints().clear();
				index++;
				if (log.isDebugEnabled() && System.currentTimeMillis() - st > 10000)
				{
					st = System.currentTimeMillis();
					log.debug("Adjust cluster center:" + ((float)index / (float)clustersTemp.size() * 100) + "%");
				}
			}
			clustersTemp.clear();
			mergeCluster();
			index = 0;
			for (ClusterPoint<T> point : pointsCache)
			{
				clusterScanner.scanCluster(clusters, point);
				if (log.isDebugEnabled() && System.currentTimeMillis() - st > 10000)
				{
					st = System.currentTimeMillis();
					log.debug("Repeat add point percent:" + ((float)index / (float)pointsCache.size() * 100) + "%");
				}
			}
			if (cohesiveValue == lastCohesiveValue)
				break;
			lastCohesiveValue = cohesiveValue;
		}
		return System.currentTimeMillis() - rt;
	}
	
	private void mergeCluster()
	{
		int beforeClusterCount = clusters.size();
		List<Cluster<T>> clustersTemp = new ArrayList<Cluster<T>>(clusters.size());
		for (Cluster<T> cluster : clusters)
		{
			ClusterPoint<T> point = cluster.getCenter();
			Cluster<T> maxCluster = null;
			double matchRate = 0;
			for (Cluster<T> temp : clustersTemp)
			{
				double rate = distancePoints(point, temp.getCenter());
				if (maxCluster == null || distance.compare(rate, matchRate))
				{
					matchRate = rate;
					maxCluster = temp;
				}
			}
			if (maxCluster != null && distance.compare(matchRate, config.getMergeSimilar()))
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
			log.debug("Merge cluster before:" + beforeClusterCount + " after:" + clusters.size());
		}
	}
	
	private double adjustCenter(Cluster<T> cluster)
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
			if (maxPoint == null || distance.compare(sum, maxRate))
			{
				maxPoint = point;
				maxRate = sum;
			}
		}
		double cohesiveValue = maxRate; 
		if (!maxPoint.equals(cluster.getCenter()))
		{
			cluster.getCenter().setCenterPoint(false);
			cluster.getPoints().add(cluster.getCenter());
			cluster.getPoints().remove(maxPoint);
			cluster.setCenter(maxPoint);
			maxPoint.setCenterPoint(true);
		}
		return cohesiveValue;
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
