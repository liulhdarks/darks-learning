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
package darks.learning.cluster.visual;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import javax.swing.JFrame;

import darks.learning.cluster.Cluster;
import darks.learning.cluster.ClusterPoint;


public class ClusterVisualization
{

	static List<Color> colors = new ArrayList<Color>();
	
	static Random random = new Random(25214903917L);
	
	static
	{
		colors.add(Color.RED);
		colors.add(Color.BLUE);
		colors.add(Color.GREEN);
		colors.add(Color.YELLOW);
		colors.add(Color.BLACK);
		colors.add(Color.CYAN);
		colors.add(Color.MAGENTA);
		colors.add(Color.ORANGE);
		colors.add(Color.PINK);
	};
	
	int showWidth = 480;
	
	int showHeight = 360;
	
	public ClusterVisualization()
	{
		
	}
	
	
	public ClusterVisualization(int showWidth, int showHeight)
	{
		super();
		this.showWidth = showWidth;
		this.showHeight = showHeight;
	}



	public void visual(Collection<Cluster<Point>> clusters)
	{
		VisualizationFrame frame = new VisualizationFrame(clusters);
		frame.setSize(showWidth, showHeight);
		frame.setVisible(true);
	}
	
	class VisualizationFrame extends JFrame
	{
		
		private static final long serialVersionUID = 5857398390001370950L;
		
		Collection<Cluster<Point>> clusters;
		
		
		public VisualizationFrame(Collection<Cluster<Point>> clusters)
		{
			this.clusters = clusters;
			if (clusters.size() > colors.size())
			{
				Set<Color> clrSet = new HashSet<Color>(colors);
				while (clusters.size() > colors.size())
				{
					Color color = new Color(random.nextInt(256), random.nextInt(256), random.nextInt(256));
					if (!clrSet.contains(color))
					{
						colors.add(color);
						clrSet.add(color);
					}
				}
			}
			setDefaultCloseOperation(DISPOSE_ON_CLOSE);
			repaint();
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public void paint(Graphics g)
		{
			super.paint(g);
			g.setColor(Color.WHITE);
			g.fillRect(0, 0, getWidth(), getHeight());
			int index = 0;
			for (Cluster<Point> cluster : clusters)
			{
				Color color = colors.get(index++);
				g.setColor(color);
				ClusterPoint<Point> center = cluster.getCenter();
				for (ClusterPoint<Point> point : cluster.getPoints())
				{
					g.fillOval(point.getObject().x, point.getObject().y, 3, 3);
				}
				g.fillOval(center.getObject().x, center.getObject().y, 8, 8);
			}
		}
		
		
	}
}
