package darks.learning.common.utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;

public final class IOUtils
{

	private IOUtils()
	{

	}

	public static void closeStream(Reader reader)
	{
		try
		{
			if (reader != null)
			{
				reader.close();
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void closeStream(Writer writer)
	{
		try
		{
			if (writer != null)
			{
				writer.close();
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void closeStream(InputStream ins)
	{
		try
		{
			if (ins != null)
			{
				ins.close();
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void closeStream(OutputStream out)
	{
		try
		{
			if (out != null)
			{
				out.close();
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

}
