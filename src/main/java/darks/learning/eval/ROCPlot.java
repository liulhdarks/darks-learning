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

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;

import javax.swing.JFrame;
import javax.swing.JPanel;

import darks.learning.eval.ROC.PlotLine;

public class ROCPlot
{
	
	private static final int PADDING = 20;

	ROC roc;
	
	JFrame frame = null;
	
	Double fitK = null;
	
	Integer evalLineId = null;
	
	public ROCPlot()
	{
		
	}

	public ROCPlot(ROC roc)
	{
		this.roc = roc;
	}

	public ROCPlot(ROC roc, int evalLineId, double fitK)
	{
		this.roc = roc;
		this.fitK = fitK;
		this.evalLineId = evalLineId;
	}
	
	public void initialize()
	{
		frame = new JFrame("Aim For the Center"); 
		frame.setTitle("ROC Eval Plot");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new ROCPanel();  
        frame.add(panel, BorderLayout.CENTER);
        frame.setSize(300 + PADDING * 2, 300 + PADDING * 2);
	}
	
	public void show()
	{
		frame.setVisible(true);  
	}
	
	class ROCPanel extends JPanel
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = 4723392521542069374L;

		@Override
		protected void paintComponent(Graphics graphic)
		{
			Graphics2D g = (Graphics2D)graphic;
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			drawBack(g);
			drawLines(g);
			if (evalLineId != null && fitK != null)
			{
				drawEstimate(g);
			}
		}
		
		private void drawEstimate(Graphics g)
		{
			g.setColor(Color.GRAY);
			int w = this.getWidth();
			int h = this.getHeight();
			int hpad = h - PADDING * 2;
			int wpad = w - PADDING * 2;
			PlotLine plotLine = roc.plotsMap.get(evalLineId);
			ROCPoint pt = roc.estimate(plotLine, fitK);
			double b = roc.getOffsetY(pt, fitK);
			int startX = PADDING;
			int startY = (int)(hpad * (1 - b)) + PADDING;
			int endX = (int)(wpad * ((1 -b) / fitK)) + PADDING;
			int endY = PADDING;
			g.drawLine(startX, startY, endX, endY);
			g.setColor(Color.BLACK);
			g.drawString("最优结果:" + pt.label + " FPR:" + pt.fpr + " TPR:" + pt.tpr, PADDING, 15);
		}
		
		private void drawLines(Graphics g)
		{
			int w = this.getWidth();
			for (PlotLine line : roc.plotsMap.values())
			{
				g.setColor(Color.RED);
				Point zeroPoint = convert(line.zeroPlot);
				Point lastPoint = zeroPoint;
				ROCPoint lastPlot = line.zeroPlot;
				while (lastPlot != null)
				{
					lastPlot = lastPlot.next;
					if (lastPlot != null)
					{
						Point curPoint = convert(lastPlot);
						g.fillOval(curPoint.x - 3, curPoint.y - 1, 5, 5);
						g.drawLine(lastPoint.x, lastPoint.y, curPoint.x, curPoint.y);
						g.drawString(lastPlot.label, curPoint.x - 3, curPoint.y + 13);
						lastPoint = curPoint;
					}
				}
				g.drawLine(lastPoint.x, lastPoint.y, w - PADDING, PADDING);
				for (ROCPoint pt : line.excludePoints)
				{
					Point curPoint = convert(pt);
					g.fillOval(curPoint.x - 3, curPoint.y - 3, 5, 5);
					g.drawString(pt.label, curPoint.x - 3, curPoint.y + 13);
				}
			}
		}
		
		private Point convert(ROCPoint point)
		{
			int w = this.getWidth();
			int h = this.getHeight();
			int wpad = w - PADDING * 2;
			int hpad = h - PADDING * 2;
			int x = (int)(wpad * point.fpr) + PADDING;
			int y = (int)(hpad * (1 - point.tpr)) + PADDING;
			return new Point(x, y);
		}
		
		private void drawBack(Graphics g)
		{
			int w = this.getWidth();
			int h = this.getHeight();
			g.setColor(Color.WHITE);
			g.fillRect(0, 0, w, h);
			g.setColor(Color.BLACK);
			g.drawLine(PADDING, h - PADDING, w - PADDING, h - PADDING);
			g.drawLine(PADDING, h - PADDING, PADDING, PADDING);
			g.drawLine(PADDING, h - PADDING, w - PADDING, PADDING);
			g.drawLine(w - PADDING, h - PADDING, w - PADDING, PADDING);
			g.setColor(Color.GRAY);
			g.drawLine(PADDING, PADDING, w - PADDING, PADDING);
			g.drawLine(PADDING, PADDING, w - PADDING, h - PADDING);
			g.setColor(Color.BLACK);
			int wpad = w - PADDING * 2;
			int hpad = h - PADDING * 2;
			for (float p = 0.1f; p <= 1; p += 0.1f)
			{
				int x = (int)(wpad * p) + PADDING;
				int y = (int)(hpad * (1 - p)) + PADDING;
				g.fillOval(x, h - PADDING, 3, 3);
				g.drawString(String.format("%.1f", p), x - 5, h - PADDING + 14);
				g.fillOval(PADDING, y, 3, 3);
				g.drawString(String.format("%.1f", p), PADDING - 17, y + 8);
			}
		}
		
	}
}
