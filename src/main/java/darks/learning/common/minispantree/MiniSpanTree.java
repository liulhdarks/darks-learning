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

import java.util.List;

import darks.learning.common.blas.Matrix;

/**
 * Mini-span tree abstract class
 * 
 * @author lihua.llh
 *
 */
public abstract class MiniSpanTree<T, E>
{
	
	protected List<? extends GraphNode<T>> nodes;
	
	protected Matrix<? extends GraphEdge<E>> edges;

	/**
	 * Initialize graph
	 * 
	 * @param builder
	 */
	public void initialize(GraphBuilder<T, E> builder)
	{
		buildGraph(builder);
	}
	
	protected void buildGraph(GraphBuilder<T, E> builder)
	{
		nodes = builder.buildNodes();
		edges = builder.buildEdges();
	}
	
	/**
	 * Execute to build mini-span tree
	 */
	public void buildTree()
	{
		buildTree(0);
	}
	
	/**
	 * Execute to build mini-span tree from start position
	 * 
	 * @param startIndex Start position
	 */
	public abstract void buildTree(int startIndex);
	
	/**
	 * Get result nodes index
	 * @return Nodes index list
	 */
	public abstract List<Integer> getResultNodesIndex();
	
	/**
	 * Get result nodes
	 * @return Result nodes
	 */
	public abstract List<? extends GraphNode<T>> getResultNodes();
	
	/**
	 * Get result edges
	 * @return Result edge
	 */
	public abstract List<? extends GraphEdge<E>> getResultEdges();
}
