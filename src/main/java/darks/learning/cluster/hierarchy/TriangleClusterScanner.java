package darks.learning.cluster.hierarchy;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig;
import darks.learning.cluster.ClusterPoint;
import darks.learning.common.basic.KeyValue;
import darks.learning.distance.Distance;

public class TriangleClusterScanner<T> extends ClusterScanner<T>
{

	Set<Cluster<T>> cacheClusters = new HashSet<Cluster<T>>();

	int skipCount = 0;
	
	public TriangleClusterScanner(Distance<T> distance, ClusterConfig config)
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
		if (point.isCenterPoint())
			return System.currentTimeMillis() - st;
		if (clusters.isEmpty())
		{
			addNewCluster(clusters, point);
			return System.currentTimeMillis() - st;
		}
		Cluster<T> minCluster = null;
		double minDistance = 0;
		Cluster<T> keepMinCluster = null;
		double keepMinDistance = 0;
		Cluster<T> cluster = clusters.get(0);
		if (cluster.getLowCluster() == null)
			computeClusterDistance(clusters, cluster);
		double d = distance.distance(cluster.getCenter().getObject(), point.getObject());
		if (keepMinCluster == null || distance.compare(d, keepMinDistance))
		{
			keepMinDistance = d;
			keepMinCluster = cluster;
		}
		if (d * 2 <= cluster.getLowDistance())
		{
			minCluster = cluster;
			minDistance = d;
		}
		else
		{
//			Map<Cluster<T>, Double> cmList = new HashMap<Cluster<T>, Double>();
			Set<KeyValue<Cluster<T>, Double>> cmSet = cluster.getClusterSet().headSet(new KeyValue<Cluster<T>, Double>(cluster, d * 2), false);
			for (KeyValue<Cluster<T>, Double> entry : cmSet)
			{
				Cluster<T> cm = entry.getKey();
//					boolean skip = false;
//					for (Entry<Cluster<T>, Double> cmEntry : cmList.entrySet())
//					{
//						Cluster<T> cm2 = cmEntry.getKey();
//						double cmCd = cmEntry.getValue();
//						Double d2 = cm2.getClusterMap().get(cm);
//						if (d2 != null && cmCd * 2 <= d2)
//							skip = true;
//					}
//					if (skip)
//						continue;
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
					minDistance = cd;
					break;
				}
				else
				{
//					cmList.put(cm, cd);
				}
			}
		}
		if (minCluster == null)
		{
			minCluster = keepMinCluster;
			minDistance = keepMinDistance;
		}
		if (minCluster != null)
		{
			minCluster.getPoints().add(point);
		}
		if (minCluster != null && distance.compare(minDistance, config.getMinSimilar()))
		{
			point.setCenterPoint(false);
			point.setSimilar(minDistance);
			minCluster.getPoints().add(point);
		}
		else
		{
			addNewCluster(clusters, point);
		}
		return System.currentTimeMillis() - st;
	}

	
	private void computeClusterDistance(Collection<Cluster<T>> clusters, Cluster<T> cluster)
	{
		double lowD = 0;
		Cluster<T> lowCluster = null;
		for (Cluster<T> target : clusters)
		{
			if (cluster.equals(target))
				continue;
			double d = distance.distance(target.getCenter().getObject(), cluster.getCenter().getObject());
			if (lowCluster == null || distance.compare(d, lowD))
			{
				lowD = d;
				lowCluster = target;
			}
			cluster.putCloestCluster(target, d);
			target.putCloestCluster(cluster, d);
			if (target.getLowCluster() != null)
			{
				if (distance.compare(d, target.getLowDistance()))
				{
					target.setLowCluster(cluster);
					target.setLowDistance(d);
				}
			}
		}
		cluster.setLowCluster(lowCluster);
		cluster.setLowDistance(lowD);
		cacheClusters.add(cluster);
	}
	
	private void addNewCluster(List<Cluster<T>> clusters, ClusterPoint<T> point)
	{
		Cluster<T> newCluster = new Cluster<T>();
		newCluster.setCenter(point);
		point.setCenterPoint(true);
		clusters.add(newCluster);
		
		for (Cluster<T> target : cacheClusters)
		{
			if (newCluster.equals(target))
				continue;
			double d = distance.distance(target.getCenter().getObject(), newCluster.getCenter().getObject());
			target.putCloestCluster(newCluster, d);
			if (target.getLowCluster() != null)
			{
				if (distance.compare(d, target.getLowDistance()))
				{
					target.setLowCluster(newCluster);
					target.setLowDistance(d);
				}
			}
		}
	}
}
