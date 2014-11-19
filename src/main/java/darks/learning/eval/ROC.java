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
package darks.learning.eval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import darks.learning.exceptions.EvaluationException;

/**
 * Receiver operating characteristic curve
 * 
 * @author Darks.Liu
 *
 */
public class ROC
{
	
	Map<Integer, PlotLine> plotsMap = new HashMap<Integer, PlotLine>();
	
	/**
	 * Add ROC line
	 * @param lineId Line id
	 * @param name Line name
	 */
	public void addLine(int lineId, String name)
	{
		PlotLine plotLine = plotsMap.get(lineId);
		if (plotLine == null)
		{
			plotLine = new PlotLine(name);
			plotsMap.put(lineId, plotLine);
		}
	}
	
	/**
	 * Add point for ROC line 
	 * @param lineId Line id
	 * @param tpr TPR
	 * @param fpr FPR
	 * @param label Point label
	 */
	public void addPoint(int lineId, double tpr, double fpr, String label)
	{
		PlotLine plotLine = plotsMap.get(lineId);
		if (plotLine == null)
		{
			throw new EvaluationException("Cannot find plot line by line id " + lineId);
		}
		plotLine.plots.add(new ROCPoint(tpr, fpr, label));
	}
	
	/**
	 * Evaluate ROC line's optimal point by slop K value.
	 * @param lineId Line id
	 * @param fitK Slop K value
	 * @return Optimal point
	 */
	public ROCPoint eval(int lineId, double fitK)
	{
		PlotLine plotLine = plotsMap.get(lineId);
		if (plotLine == null)
		{
			throw new EvaluationException("Cannot find plot line by line id " + lineId);
		}
		prepareLine(plotLine);
		return estimate(plotLine, fitK);
	}
	
	public void evalPlot(int lineId, double fitK)
	{
		PlotLine plotLine = plotsMap.get(lineId);
		if (plotLine == null)
		{
			throw new EvaluationException("Cannot find plot line by line id " + lineId);
		}
		prepareLine(plotLine);
		ROCPlot plot = new ROCPlot(this, lineId, fitK);
		plot.initialize();
		plot.show();
	}
	
	public void showPlot()
	{
		ROCPlot plot = new ROCPlot(this);
		plot.initialize();
		plot.show();
	}
	
	private void prepareLine(PlotLine plotLine)
	{
		plotLine.zeroPlot = new ROCPoint(0, 0, "0", false);
		ROCPoint onePoint = new ROCPoint(1, 1, "1", false);
		List<ROCPoint> points = new ArrayList<ROCPoint>(plotLine.plots);
		points.add(onePoint);
		ROCPoint lastPlot = plotLine.zeroPlot;
		for (ROCPoint plot : points)
		{
			double k = gradient(lastPlot, plot);
			if (lastPlot.prev == null)
			{
				lastPlot.next = plot;
				plot.prev = lastPlot;
			}
			else
			{
				ROCPoint prev = lastPlot;
				while (prev != null)
				{
					if (k <= prev.k)
					{
						break;
					}
					if (prev.prev == null)
					{
						break;
					}
					plotLine.excludePoints.add(prev);
					prev = prev.prev;
					k = gradient(prev, plot);
				}
				if (!plot.valid)
				{
					prev.next = null;
				}
				else
				{
					prev.next = plot;
					plot.prev = prev;
				}
			}
			plot.k = k;
			lastPlot = plot;
		}
	}
	
	ROCPoint estimate(PlotLine plotLine, double fitK)
	{
		ROCPoint lastPlot = plotLine.zeroPlot;
		double maxY = Double.MIN_VALUE;
		ROCPoint maxPlot = null;
		while (lastPlot != null)
		{
			lastPlot = lastPlot.next;
			if (lastPlot != null)
			{
				double offset = getOffsetY(lastPlot, fitK);
				if (offset > maxY)
				{
					maxPlot = lastPlot;
					maxY = offset;
				}
			}
		}
		return maxPlot;
	}
	
	double gradient(ROCPoint plot1, ROCPoint plot2)
	{
		return (plot2.tpr - plot1.tpr) / (plot2.fpr - plot1.fpr);
	}
	
	double getOffsetY(ROCPoint plot, double fitK)
	{
		return plot.tpr - plot.fpr * fitK;
	}
	
	class PlotLine
	{
		String name;
		
		Set<ROCPoint> plots = new TreeSet<ROCPoint>();
		
		ROCPoint zeroPlot;
		
		List<ROCPoint> excludePoints = new ArrayList<ROCPoint>();

		public PlotLine(String name)
		{
			super();
			this.name = name;
		}
		
	}
}
