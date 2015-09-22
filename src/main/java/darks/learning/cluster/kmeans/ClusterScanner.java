package darks.learning.cluster.kmeans;

import java.util.List;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterPoint;
import darks.learning.distance.Distance;

/**
 * Scan cluster
 * @author lihua.llh
 *
 */
public abstract class ClusterScanner<T>
{
	protected Distance<T> distance;
	
	public ClusterScanner(Distance<T> distance)
	{
		this.distance = distance;
	}

	public abstract long scanCluster(List<Cluster<T>> clusters, List<ClusterPoint<T>> points);
	
}
