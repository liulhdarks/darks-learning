package darks.learning.test.common;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import darks.learning.common.blas.DenseMatrix;
import darks.learning.common.blas.Matrix;
import darks.learning.common.blas.SymmetricMatrix;
import darks.learning.common.minispantree.DirectPrimMiniSpanTree;
import darks.learning.common.minispantree.GraphBuilder;
import darks.learning.common.minispantree.GraphEdge;
import darks.learning.common.minispantree.GraphNode;
import darks.learning.common.minispantree.MiniSpanTree;
import darks.learning.common.minispantree.PrimMiniSpanTree;

public class MiniSpanTreeTest
{
	
	@Test
	public void testPrim()
	{
		MiniSpanTree<String, String> tree = new PrimMiniSpanTree<String, String>();
		tree.initialize(new GraphBuilder<String, String>()
		{
			
			@Override
			public List<? extends GraphNode<String>> buildNodes()
			{
				List<GraphNode<String>> result = new ArrayList<GraphNode<String>>();
				result.add(new SpanTreeNode("A"));
				result.add(new SpanTreeNode("B"));
				result.add(new SpanTreeNode("C"));
				result.add(new SpanTreeNode("D"));
				result.add(new SpanTreeNode("E"));
				result.add(new SpanTreeNode("F"));
				result.add(new SpanTreeNode("G"));
				return result;
			}
			
			@Override
			public SymmetricMatrix<? extends GraphEdge<String>> buildEdges()
			{
				SymmetricMatrix<GraphEdge<String>> edges = new SymmetricMatrix<GraphEdge<String>>(7);
				edges.put(0, 1, new SpanTreeEdge(0, 1, 7));
				edges.put(0, 3, new SpanTreeEdge(0, 3, 5));
				edges.put(1, 3, new SpanTreeEdge(1, 3, 9));
				edges.put(1, 2, new SpanTreeEdge(1, 2, 8));
				edges.put(1, 4, new SpanTreeEdge(1, 4, 7));
				edges.put(2, 4, new SpanTreeEdge(2, 4, 5));
				edges.put(3, 4, new SpanTreeEdge(3, 4, 15));
				edges.put(3, 5, new SpanTreeEdge(3, 5, 6));
				edges.put(4, 5, new SpanTreeEdge(4, 5, 8));
				edges.put(4, 6, new SpanTreeEdge(4, 6, 9));
				edges.put(5, 6, new SpanTreeEdge(5, 6, 11));
				return edges;
			}
		});
		tree.buildTree(0);
		System.out.println(tree.getResultNodesIndex());
		for (GraphEdge<String> edge : tree.getResultEdges())
		{
			System.out.println(edge);
		}
	}
	

	@Test
	public void testDirectPrim()
	{
		MiniSpanTree<String, String> tree = new DirectPrimMiniSpanTree<String, String>();
		tree.initialize(new GraphBuilder<String, String>()
		{
			
			@Override
			public List<? extends GraphNode<String>> buildNodes()
			{
				List<GraphNode<String>> result = new ArrayList<GraphNode<String>>();
				result.add(new SpanTreeNode("A"));
				result.add(new SpanTreeNode("B"));
				result.add(new SpanTreeNode("C"));
				result.add(new SpanTreeNode("D"));
				result.add(new SpanTreeNode("E"));
				result.add(new SpanTreeNode("F"));
				result.add(new SpanTreeNode("G"));
				return result;
			}
			
			@Override
			public Matrix<? extends GraphEdge<String>> buildEdges()
			{
				Matrix<GraphEdge<String>> edges = new DenseMatrix<GraphEdge<String>>(7);
				edges.put(1, 0, new SpanTreeEdge(0, 1, 7));
				edges.put(3, 0, new SpanTreeEdge(0, 3, 5));
				edges.put(3, 1, new SpanTreeEdge(1, 3, 9));
				edges.put(2, 1, new SpanTreeEdge(1, 2, 8));
				edges.put(4, 1, new SpanTreeEdge(1, 4, 7));
				edges.put(4, 2, new SpanTreeEdge(2, 4, 5));
				edges.put(4, 3, new SpanTreeEdge(3, 4, 15));
				edges.put(5, 3, new SpanTreeEdge(3, 5, 6));
				edges.put(5, 4, new SpanTreeEdge(4, 5, 8));
				edges.put(6, 4, new SpanTreeEdge(4, 6, 9));
				edges.put(6, 5, new SpanTreeEdge(5, 6, 11));
				
				edges.put(0, 1, new SpanTreeEdge(1, 0, 7));
				edges.put(0, 3, new SpanTreeEdge(3, 0, 5));
				edges.put(1, 3, new SpanTreeEdge(3, 1, 9));
				edges.put(1, 2, new SpanTreeEdge(2, 1, 8));
				edges.put(1, 4, new SpanTreeEdge(4, 1, 7));
				edges.put(2, 4, new SpanTreeEdge(4, 2, 5));
				edges.put(3, 4, new SpanTreeEdge(4, 3, 15));
				edges.put(3, 5, new SpanTreeEdge(5, 3, 6));
				edges.put(4, 5, new SpanTreeEdge(5, 4, 8));
				edges.put(4, 6, new SpanTreeEdge(6, 4, 9));
				edges.put(5, 6, new SpanTreeEdge(6, 5, 11));
				return edges;
			}
		});
		tree.buildTree(0);
		System.out.println(tree.getResultNodesIndex());
		for (GraphEdge<String> edge : tree.getResultEdges())
		{
			System.out.println(edge);
		}
	}

	class SpanTreeNode extends GraphNode<String>
	{

		public SpanTreeNode()
		{
			super();
		}

		public SpanTreeNode(String name)
		{
			super(name);
		}
		
	}

	class SpanTreeEdge extends GraphEdge<String>
	{

		public SpanTreeEdge()
		{
			super();
		}

		public SpanTreeEdge(int from, int to, double weight)
		{
			super(from, to, weight);
		}
		
	}
}
