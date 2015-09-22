package darks.learning.cluster.kmeans;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

/**
 * Scan cluster points with triangle inequality
 * 
 * @author lihua.llh
 *
 */
public class TriangleClusterScanner<T> extends ClusterScanner<T>
{

	public TriangleClusterScanner(Distance<T> distance)
	{
		super(distance);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public long scanCluster(List<Cluster<T>> clusters, List<ClusterPoint<T>> points)
	{
		long st = System.currentTimeMillis();
		for (ClusterPoint<T> point : points)
		{
			if (point.isCenterPoint())
				continue;
			List<Cluster<T>> CMList = new LinkedList<Cluster<T>>();
			Cluster<T> minCluster = null;
			Cluster<T> keepMinCluster = null;
			double keepMinDistance = 0;
			Cluster<T> cluster = clusters.get(0);
			if (cluster.getLowCluster() == null)
				computeClusterDistance(clusters, cluster);
			double d = distance.distance(cluster.getCenter().getObject(), point.getObject());
			if (keepMinCluster == null || d < keepMinDistance)
			{
				keepMinDistance = d;
				keepMinCluster = cluster;
			}
			if (d * 2 <= cluster.getLowDistance())
			{
				minCluster = cluster;
			}
			else
			{
				for (Entry<Cluster<T>, Double> entry : cluster.getClusterMap().entrySet())
				{
					if (d * 2 > entry.getValue())
					{
						CMList.add(entry.getKey());
					}
				}
				Set<Cluster<T>> skip = new HashSet<Cluster<T>>();
				for (Cluster<T> cm : CMList)
				{
					if (skip.contains(cm))
						continue;
					if (cm.getLowCluster() == null)
						computeClusterDistance(clusters, cm);
					double cd = distance.distance(cm.getCenter().getObject(), point.getObject());
					if (keepMinCluster == null || cd < keepMinDistance)
					{
						keepMinDistance = cd;
						keepMinCluster = cm;
					}
					if (cd * 2 <= cm.getLowDistance())
					{
						minCluster = cm;
						break;
					}
					else
					{
						for (Cluster<T> cm2 : CMList)
						{
							Double d2 = cm.getClusterMap().get(cm2);
							if (d2 != null && cd * 2 <= d2)
							{
								skip.add(cm2);
							}
						}
					}
				}
			}
			if (minCluster == null)
				minCluster = keepMinCluster;
			if (minCluster != null)
			{
				minCluster.getPoints().add(point);
			}
		}
		return System.currentTimeMillis() - st;
	}
	
	private void computeClusterDistance(List<Cluster<T>> clusters, Cluster<T> cluster)
	{
		double lowD = 0;
		Cluster<T> lowCluster = null;
		for (Cluster<T> target : clusters)
		{
			if (cluster.equals(target))
				continue;
			double d = distance.distance(target.getCenter().getObject(), cluster.getCenter().getObject());
			if (lowCluster == null || d < lowD)
			{
				lowD = d;
				lowCluster = target;
			}
			cluster.getClusterMap().put(target, d);
		}
		cluster.setLowCluster(lowCluster);
		cluster.setLowDistance(lowD);
	}

}
