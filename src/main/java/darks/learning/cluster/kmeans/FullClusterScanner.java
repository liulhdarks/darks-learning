package darks.learning.cluster.kmeans;

import java.util.List;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

/**
 * Scan cluster points fully
 * @author lihua.llh
 *
 */
public class FullClusterScanner<T> extends ClusterScanner<T>
{

	public FullClusterScanner(Distance<T> distance)
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
			double minD = 0;
			Cluster<T> minCluster = null;
			for (Cluster<T> cluster : clusters)
			{
				double d = distance.distance(cluster.getCenter().getObject(), point.getObject());
				if (minCluster == null || d < minD)
				{
					minD = d;
					minCluster = cluster;
				}
			}
			minCluster.getPoints().add(point);
		}
		return System.currentTimeMillis() - st;
	}

}
