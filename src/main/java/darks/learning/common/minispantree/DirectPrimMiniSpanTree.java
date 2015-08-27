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
package darks.learning.common.minispantree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Build minimum span tree by Prim algorithm with direction
 *  
 * @author lihua.llh
 *
 */
public class DirectPrimMiniSpanTree<T, E> extends MiniSpanTree<T, E>
{
	
	LinkedHashSet<Integer> selectNodes;
	
	List<GraphEdge<E>> targetEdges;
	
	double[] minCost;
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void initialize(GraphBuilder<T, E> builder)
	{
		super.initialize(builder);
		selectNodes = new LinkedHashSet<Integer>();
		targetEdges = new LinkedList<GraphEdge<E>>();
		minCost = new double[nodes.size()];
		Arrays.fill(minCost, Double.MAX_VALUE / 3);
		selectNodes.addAll(builder.getSelectNodes());
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void buildTree(int startIndex)
	{
		GraphEdge<E> firstEdge = initStartEdge(startIndex);
		PriorityQueue<GraphEdge<E>> queue = new PriorityQueue<GraphEdge<E>>();
		queue.add(firstEdge);
		while (!queue.isEmpty())
		{
			GraphEdge<E> edge = queue.poll();
			if (edge == null)
				continue;
			int id = edge.from;
			if (selectNodes.contains(id) || edge.weight > minCost[id])
				continue;
			selectNodes.add(id);
			targetEdges.add(edge);
			for (int c = 0; c < edges.columns(); c++)
			{
				GraphEdge<E> fromEdge = edges.get(id, c);
				if (fromEdge != null)
				{
					if (minCost[fromEdge.from] > fromEdge.getWeight())
	                {
						minCost[fromEdge.from] = fromEdge.getWeight();
	                    queue.add(fromEdge);
	                }
				}
			}
		}
	}
	
	private GraphEdge<E> initStartEdge(int startIndex)
	{
		GraphEdge<E> firstEdge = null;
		double minCost = Double.MAX_VALUE;
		for (int c = 0; c < edges.columns(); c++)
		{
			GraphEdge<E> edge = edges.get(startIndex, c);
			if (edge != null)
			{
				if (minCost > edge.weight)
				{
					minCost = edge.weight;
					firstEdge = edge;
				}
			}
		}
		selectNodes.add(startIndex);
		return firstEdge;
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Integer> getResultNodesIndex()
	{
		return new ArrayList<Integer>(selectNodes);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<? extends GraphNode<T>> getResultNodes()
	{
		List<GraphNode<T>> result = new ArrayList<GraphNode<T>>(selectNodes.size());
		for (Integer index : selectNodes)
		{
			result.add(nodes.get(index));
		}
		return result;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<? extends GraphEdge<E>> getResultEdges()
	{
		return targetEdges;
	}

	
}
