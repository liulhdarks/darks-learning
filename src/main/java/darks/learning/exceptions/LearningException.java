package darks.learning.exceptions;

public class LearningException extends RuntimeException
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7941341353467687996L;

	public LearningException()
	{
		super();
	}

	public LearningException(String message, Throwable cause)
	{
		super(message, cause);
	}

	public LearningException(String message)
	{
		super(message);
	}

	public LearningException(Throwable cause)
	{
		super(cause);
	}

	
}
