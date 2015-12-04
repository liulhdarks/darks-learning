package darks.learning.test.cluster;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterConfig.OptimizeType;
import darks.learning.cluster.ClusterExecutor;
import darks.learning.cluster.hierarchy.DynamicalCluster;
import darks.learning.cluster.visual.ClusterVisualization;
import darks.learning.distance.Distance;

public class PointClusterTest
{

	@Test
	public void testPointCluster()
	{
		ClusterExecutor<Point> cluster = new DynamicalCluster<Point>(new Distance<Point>(Distance.TYPE_DISTANCE)
		{
			@Override
			public double distance(Point a, Point b)
			{
				return Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
			}
		});
		cluster.config.setIterateCount(100);
		cluster.config.setClusterCount(5000);
		cluster.config.setOptimizeType(OptimizeType.NONE);
		cluster.config.setMinSimilar(10);
		cluster.config.setMergeSimilar(15);
		
		int width = 800;
		int height = 600;
		cluster.cluster(emitPoints(true, width, height));
		Collection<Cluster<Point>> clusters = cluster.getClusters();
		ClusterVisualization visual = new ClusterVisualization(width, height);
		visual.visual(clusters);
		Assert.assertNotNull(clusters);
	}
	
	private List<Point> emitPoints(boolean allowRepeat, int width, int height)
	{
		Random rand = new Random(111111111111l);
		List<Point> data = new ArrayList<Point>();
		Set<Integer> pointSet = new HashSet<Integer>();
		for (int i = 0; i < 100; i++)
		{
			int key = 0;
			int x = 0;
			int y = 0;
			if (allowRepeat)
			{
				x = rand.nextInt(width); 
				y = rand.nextInt(height);
				key = y * height + x;
				if (!pointSet.contains(key))
				{
					data.add(new Point(x, y));
					pointSet.add(key);
				}
			}
			else
			{
				do
				{
					x = rand.nextInt(width); 
					y = rand.nextInt(height);
					key = y * height + x;
				} while (pointSet.contains(key));
				pointSet.add(key);
				data.add(new Point(x, y));
			}
		}
		return data;
	}
	    
}
