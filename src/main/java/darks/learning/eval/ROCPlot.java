/**
 * 
 * Copyright 2014 The Darks Learning Project (Liu lihua)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
package darks.learning.eval;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import darks.learning.eval.ROC.PlotLine;

/**
 * Receiver operating characteristic curve plot
 * 
 * <pre>
 * lineId = 1;
 * ROC roc = new ROC();
 * roc.addLine(lineId, &quot;demo&quot;);
 * roc.addPoint(lineId, 0.2, 0.1, &quot;v1&quot;);
 * roc.addPoint(lineId, 0.4, 0.4, &quot;v2&quot;);
 * roc.addPoint(lineId, 0.3, 0.6, &quot;v3&quot;);
 * roc.addPoint(lineId, 0.5, 0.7, &quot;v4&quot;);
 * roc.addPoint(lineId, 0.3, 0.2, &quot;v5&quot;);
 * roc.addPoint(lineId, 0.6, 0.8, &quot;v6&quot;);
 * roc.eval(lineId, 1.0);
 * roc.showPlot();
 * // roc.evalPlot(lineId, 1.0);
 * </pre>
 * 
 * @author Darks.Liu
 * 
 */
public class ROCPlot
{

	private static final int PADDING = 20;
	
	private static int fontSize = 15;

	static Color[] colors = new Color[] { Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW,
			Color.ORANGE, new Color(240, 23, 224), new Color(43, 221, 221), Color.PINK,
			Color.MAGENTA, Color.CYAN };

	ROC roc;

	JFrame frame = null;

	Double fitK = 1.0;

	Integer evalLineId = null;

	boolean showExlude = false;

	JPanel panel = null;

	JPanel fpanel = null;

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
		panel = new ROCPanel();
		fpanel = new FunctionPanel();
		frame.add(panel, BorderLayout.CENTER);
		frame.add(fpanel, BorderLayout.EAST);
		frame.setSize(600, 500);
	}

	public void show()
	{
		frame.setVisible(true);
	}

	class FunctionPanel extends JPanel
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = 638343451991194184L;

		JLabel labelK;

		JLabel labelFit;

		JButton btnK;

		JTextField txtK;

		JButton btnExclude;

		JTextField txtLineId;

		JButton btnFit;

		JButton btnAUC;

		public FunctionPanel()
		{
			setLayout(new GridLayout(12, 1, 5, 0));
			labelK = new JLabel("Fit K Value:");
			add(labelK);
			txtK = new JTextField("1.0");
			add(txtK);
			btnK = new JButton("Modify K");
			add(btnK);
			labelFit = new JLabel("Fit Line ID:");
			add(labelFit);
			txtLineId = new JTextField(evalLineId == null ? "" : String.valueOf(evalLineId));
			add(txtLineId);
			btnFit = new JButton("Fit Line");
			add(btnFit);
			btnExclude = new JButton("Exclude");
			add(btnExclude);
			btnAUC = new JButton("AUC");
			add(btnAUC);
			btnK.addActionListener(new ActionListener()
			{

				@Override
				public void actionPerformed(ActionEvent e)
				{
					String k = txtK.getText().trim();
					if ("".equals(k))
					{
						k = "1.0";
						txtK.setText(k);
					}
					fitK = Double.valueOf(k);
					panel.repaint();
				}
			});
			btnExclude.addActionListener(new ActionListener()
			{

				@Override
				public void actionPerformed(ActionEvent e)
				{
					showExlude = !showExlude;
					panel.repaint();
				}
			});
			btnFit.addActionListener(new ActionListener()
			{

				@Override
				public void actionPerformed(ActionEvent e)
				{
					String lineId = txtLineId.getText().trim();
					if ("".equals(lineId))
					{
						return;
					}
					int id = Integer.parseInt(lineId);
					evalLineId = id;
					panel.repaint();
				}
			});
			btnAUC.addActionListener(new ActionListener()
			{

				@Override
				public void actionPerformed(ActionEvent e)
				{
					StringBuilder buf = new StringBuilder();
					for (PlotLine line : roc.plotsMap.values())
					{
						double auc = roc.aucValue(line.lineId);
						buf.append(line.name).append(':').append(auc).append("\r\n");
					}
					JOptionPane.showMessageDialog(null, buf.toString(), "AUC List", JOptionPane.PLAIN_MESSAGE);
				}
			});
		}
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
			Graphics2D g = (Graphics2D) graphic;
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
			if (plotLine == null)
			{
				g.drawString("Cannot find line id " + evalLineId, PADDING, 15);
				return;
			}
			ROCPoint pt = roc.estimate(plotLine, fitK);
			double b = roc.getOffsetY(pt, fitK);
			int startX = PADDING;
			int startY = (int) (hpad * (1 - b)) + PADDING;
			int endX = (int) (wpad * ((1 - b) / fitK)) + PADDING;
			int endY = PADDING;
			g.drawLine(startX, startY, endX, endY);
			g.setColor(Color.BLACK);
			g.drawString("Optimal Point:" + pt.label + " FPR:" + pt.fpr + " TPR:" + pt.tpr + "  K="
					+ String.format("%.2f", fitK), PADDING, 15);
		}

		private void drawLines(Graphics g)
		{
			int size = roc.plotsMap.size();
			int lineIndex = 0;
			int w = this.getWidth();
			int colorIndex = 0;
			for (PlotLine line : roc.plotsMap.values())
			{
				if (line.zeroPlot == null)
				{
					roc.eval(line.lineId, fitK);
				}
				if (line.zeroPlot == null)
				{
					continue;
				}
				g.setColor(colors[colorIndex]);
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
				if (showExlude)
				{
					for (ROCPoint pt : line.excludePoints)
					{
						Point curPoint = convert(pt);
						g.fillOval(curPoint.x - 3, curPoint.y - 3, 5, 5);
						g.drawString(pt.label, curPoint.x - 3, curPoint.y + 13);
					}
				}
				drawLineName(g, lineIndex++, size, line, colors[colorIndex]);
				colorIndex++;
				if (colorIndex >= colors.length)
				{
					colorIndex = 0;
				}
			}
		}
		
		private void drawLineName(Graphics g, int index, int size, PlotLine line, Color color)
		{
			String text = line.name;
			FontMetrics fm = this.getFontMetrics(getFont());
			int heightSize = Math.max(fm.getHeight(), fontSize);
			int total = size * heightSize;
			int w = this.getWidth();
			int h = this.getHeight();
			int txtw = fm.stringWidth(text);
			int y = h - PADDING - total - 5 + index * heightSize;
			int x = w - PADDING - txtw;
			g.drawString(text, x, y);
		}

		private Point convert(ROCPoint point)
		{
			int w = this.getWidth();
			int h = this.getHeight();
			int wpad = w - PADDING * 2;
			int hpad = h - PADDING * 2;
			int x = (int) (wpad * point.fpr) + PADDING;
			int y = (int) (hpad * (1 - point.tpr)) + PADDING;
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
				int x = (int) (wpad * p) + PADDING;
				int y = (int) (hpad * (1 - p)) + PADDING;
				g.fillOval(x, h - PADDING, 3, 3);
				g.drawString(String.format("%.1f", p), x - 5, h - PADDING + 14);
				g.fillOval(PADDING, y, 3, 3);
				g.drawString(String.format("%.1f", p), PADDING - 17, y + 8);
			}
		}

	}
}
