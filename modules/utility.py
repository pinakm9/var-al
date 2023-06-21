# A helper module for various sub-tasks
from time import time

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func