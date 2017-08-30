import numpy as np
from util import raiseNotDefined

def gradient(f, w):
    """YOUR CODE HERE"""
  #raiseNotDefined()
  #return np.gradient(f,w)
    small_enough_val = 0.000001
    close_w = [[w[i] for i in range(len(w))] for _ in range(len(w))]
    for i in range(len(w)):
      for j in range(len(w)):
        if i==j:
          close_w[i][j] += small_enough_val
    f_primes = [0 for _ in range(len(w))]
    for i in range(len(w)):
      f_primes[i] = (f(close_w[i])-f(w))/(small_enough_val)
    #print 'gradient result:',f_primes
    return f_primes



def sanity_check_gradient():
  """Handy function for debugging your gradient method."""
  def g(w):
    w1 = w[0]
    w2 = w[1]
    return w1 ** 3 * w2 + 3 * w1
  
  print("The print statement below should output approximately [111, 27]")
  print(gradient(g, np.array([3, 4], dtype='f')))

  def loss(self, sample, label, w):
      """sample: np.array of shape(1, numFeatures).
      label:  the correct label of the sample 
      w:      the weight vector under which to calculate loss

      Can interpret loss as the probability of sample being in the correct class when
      classified by a SigmoidNeuron. 

      For numerical accuracy reasons, the loss is expressed as 
      math.log(1/sigmoid) instead of -math.log(sigmoid) as we discussed in class.

      Do not modify this function."""
      z = np.dot(w, sample)

      if label == True:
          return math.log(1.0 + math.exp(-2*z))
      else:
          return math.log(1.0 + math.exp(2*z))