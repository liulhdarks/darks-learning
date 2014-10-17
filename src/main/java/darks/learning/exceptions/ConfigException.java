package darks.learning.exceptions;

public class ConfigException extends RuntimeException
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7941341353467687996L;

	public ConfigException()
	{
		super();
	}

	public ConfigException(String message, Throwable cause)
	{
		super(message, cause);
	}

	public ConfigException(String message)
	{
		super(message);
	}

	public ConfigException(Throwable cause)
	{
		super(cause);
	}

	
}
