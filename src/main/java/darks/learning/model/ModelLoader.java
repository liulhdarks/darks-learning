package darks.learning.model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.StringTokenizer;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import darks.learning.common.utils.IOUtils;
import darks.learning.exceptions.ModelException;

/**
 * Model loader
 * 
 * @author Darks.Liu
 * 
 */
public class ModelLoader
{

	private static Logger log = LoggerFactory.getLogger(ModelLoader.class);

	private ModelLoader()
	{

	}

	/**
	 * Load model set from specify file
	 * 
	 * @param file Source file
	 * @return Model set
	 */
	public static ModelSet loadFromFile(File file)
	{
		if (!file.exists())
		{
			throw new ModelException("Model file " + file + " does not exist.");
		}
		try
		{
			return loadFromStream(new FileInputStream(file));
		}
		catch (FileNotFoundException e)
		{
			throw new ModelException("Model file " + file + " does not exist.", e);
		}
	}

	/**
	 * Load model set from stream
	 * 
	 * @param ins {@linkplain java.io.InputStream InputStream}
	 * @return Model set
	 */
	public static ModelSet loadFromStream(InputStream ins)
	{
		BufferedReader reader = null;
		try
		{
			List<ModelRow> modelRows = new LinkedList<ModelRow>();
			reader = new BufferedReader(new InputStreamReader(ins));
			String line = null;
			int maxColumn = 0;
			int maxLabel = 0;
			int minLabel = 0;
			while ((line = reader.readLine()) != null)
			{
				StringTokenizer token = new StringTokenizer(line, " \t\n\r\f:");
				String type = token.nextToken().replace("+", "");
				int classify = (int)Double.parseDouble(type);
				maxLabel = Math.max(maxLabel, classify);
				minLabel = Math.min(minLabel, classify);
				ModelRow row = new ModelRow();
				row.setOutput(classify);
				int m = token.countTokens() / 2;
				for (int j = 0; j < m; j++)
				{
					int index = Integer.parseInt(token.nextToken());
					double value = Double.parseDouble(token.nextToken());
					row.add(index, value);
					maxColumn = Math.max(maxColumn, index);
				}
				modelRows.add(row);
			}
			return convertModelMatrix(maxColumn, maxLabel, minLabel, modelRows);
		}
		catch (Exception e)
		{
			log.error(e.getMessage(), e);
			return null;
		}
		finally
		{
			IOUtils.closeStream(reader);
		}
	}

	private static ModelSet convertModelMatrix(int maxColumn, int maxLabel, int minLabel, List<ModelRow> modelRows)
	{
		int rows = modelRows.size();
		int cols = maxColumn;
		int offset = Math.abs(minLabel);
		int r = 0;
		DoubleMatrix input = new DoubleMatrix(rows, cols);
		DoubleMatrix output = new DoubleMatrix(rows, maxLabel + offset +1);
		for (ModelRow row : modelRows)
		{
			DoubleMatrix maxtrix = new DoubleMatrix(maxColumn);
			for (Entry<Integer, Double> entry : row.data.entrySet())
			{
				int index = entry.getKey() - 1;
				double value = entry.getValue();
				if (index >= 0 && index < maxColumn)
				{
					maxtrix.put(index, value);
				}
			}
			DoubleMatrix label = new DoubleMatrix(maxLabel + offset +1);
			label.put(row.output + offset, 1.0);
			output.putRow(r, label);
			input.putRow(r++, maxtrix);
		}
		return new ModelSet(input, output);
	}

	static class ModelRow
	{
		Map<Integer, Double> data = new HashMap<Integer, Double>();

		int output;

		public void add(int index, double value)
		{
			data.put(index, value);
		}

		public double getOutput()
		{
			return output;
		}

		public void setOutput(int output)
		{
			this.output = output;
		}

		@Override
		public String toString()
		{
			return "ModelRow [data=" + data + ", output=" + output + "]";
		}

	}
}
