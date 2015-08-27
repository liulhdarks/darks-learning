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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import darks.learning.common.blas.Matrix;

public class GraphBuilder<T, E>
{
	
	List<? extends GraphNode<T>> nodes;
	
	Matrix<? extends GraphEdge<E>> edges;
	
	Set<Integer> selectNodes;
	
	public GraphBuilder()
	{
		selectNodes = new HashSet<Integer>();
	}

	public GraphBuilder(List<? extends GraphNode<T>> nodes, 
			Matrix<? extends GraphEdge<E>> edges)
	{
		this.nodes = nodes;
		this.edges = edges;
		selectNodes = new HashSet<Integer>();
	}
	
	public List<? extends GraphNode<T>> buildNodes()
	{
		return nodes;
	}
	
	public Matrix<? extends GraphEdge<E>> buildEdges()
	{
		return edges;
	}

	public Set<Integer> getSelectNodes()
	{
		return selectNodes;
	}

}
