package darks.learning.model;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

/**
 * Sample model set
 * @author Darks.Liu
 *
 */
public class ModelSet
{

	private List<String> columnNames = new ArrayList<String>();
	
	private List<String> columnLabels = new ArrayList<String>();
	
	private DoubleMatrix input;
	
	private DoubleMatrix output;
	
	public ModelSet()
	{
		
	}
	
	public ModelSet(DoubleMatrix input)
	{
		this.input = input;
	}
	
	public ModelSet(DoubleMatrix input, DoubleMatrix output)
	{
		this.input = input;
		this.output = output;
	}

	public DoubleMatrix getInput()
	{
		return input;
	}

	public void setInput(DoubleMatrix input)
	{
		this.input = input;
	}

	public DoubleMatrix getOutput()
	{
		return output;
	}

	public void setOutput(DoubleMatrix output)
	{
		this.output = output;
	}

	public List<String> getColumnNames()
	{
		return columnNames;
	}

	public List<String> getColumnLabels()
	{
		return columnLabels;
	}
	

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("===========INPUT===================\n")
                .append(getInput().toString().replaceAll(";","\n"))
                .append("\n=================OUTPUT==================\n")
                .append(getOutput().toString().replaceAll(";","\n"));
        return builder.toString();
    }
}
