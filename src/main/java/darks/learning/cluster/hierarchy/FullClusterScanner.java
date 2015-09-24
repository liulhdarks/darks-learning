package darks.learning.cluster.hierarchy;

import java.util.List;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

public class FullClusterScanner<T> extends ClusterScanner<T>
{


	public FullClusterScanner(Distance<T> distance, ClusterConfig config)
	{
		super(distance, config);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public long scanCluster(List<Cluster<T>> clusters, ClusterPoint<T> point)
	{
		long st = System.currentTimeMillis();
		Cluster<T> maxCluster = null;
		double matchRate = 0;
		for (Cluster<T> cluster : clusters)
		{
			double rate = distancePoints(point, cluster.getCenter());
			if (maxCluster == null || distance.compare(rate, matchRate))
			{
				matchRate = rate;
				maxCluster = cluster;
			}
		}
		if (maxCluster != null && distance.compare(matchRate, config.getMinSimilar()))
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
		return System.currentTimeMillis() - st;
	}
	
	private double distancePoints(ClusterPoint<T> point, ClusterPoint<T> target)
	{
		return distance.distance(point.getObject(), target.getObject()); 
	}

}
