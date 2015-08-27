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
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

/**
 * Build minimum span tree by Prim algorithm
 *  
 * @author lihua.llh
 *
 */
public class PrimMiniSpanTree<T, E> extends MiniSpanTree<T, E>
{
	
	List<Integer> selectNodes;
	
	Set<Integer> remainNodes;
	
	List<GraphEdge<E>> targetEdges;
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public void initialize(GraphBuilder<T, E> builder)
	{
		super.initialize(builder);
		selectNodes = new ArrayList<Integer>();
		remainNodes = new HashSet<Integer>();
		targetEdges = new LinkedList<GraphEdge<E>>();
		for (int i = 0 ; i < nodes.size(); i++)
		{
			remainNodes.add(i);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void buildTree(int startIndex)
	{
		selectNodes.add(startIndex);
		remainNodes.remove(startIndex);
		for (int i = 0; i < nodes.size() - 1; i++)
		{
			GraphEdge<E> minEdge = null;
			Integer newIndex = null;
			for (Integer remainIndex : remainNodes)
			{
				for (Integer selectIndex : selectNodes)
				{
					GraphEdge<E> edge = edges.get(selectIndex, remainIndex);
					if (edge != null)
					{
						if (minEdge == null || edge.getWeight() < minEdge.getWeight())
						{
							newIndex = remainIndex;
							minEdge = edge;
						}
					}
				}
			}
			if (newIndex != null && minEdge != null)
			{
				remainNodes.remove(newIndex);
				selectNodes.add(newIndex);
				targetEdges.add(minEdge);
			}
		}
	}
	
	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Integer> getResultNodesIndex()
	{
		return selectNodes;
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
