package darks.learning.cluster.hierarchy;

import java.util.List;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig;
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
	
	protected ClusterConfig config;
	
	public ClusterScanner(Distance<T> distance, ClusterConfig config)
	{
		this.distance = distance;
		this.config = config;
	}

	public abstract long scanCluster(List<Cluster<T>> clusters, ClusterPoint<T> point);
	
}
