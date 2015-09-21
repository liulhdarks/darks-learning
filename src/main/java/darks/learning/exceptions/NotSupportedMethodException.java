package darks.learning.exceptions;

public class NotSupportedMethodException extends RuntimeException
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7941341353467687996L;

	public NotSupportedMethodException()
	{
		super();
	}

	public NotSupportedMethodException(String message, Throwable cause)
	{
		super(message, cause);
	}

	public NotSupportedMethodException(String message)
	{
		super(message);
	}

	public NotSupportedMethodException(Throwable cause)
	{
		super(cause);
	}

	
}
