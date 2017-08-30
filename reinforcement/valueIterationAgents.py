# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(0,self.iterations):
          temp = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              temp[state] = 0
            else:
              maxVal = float("-inf") 
              for action in self.mdp.getPossibleActions(state):
                tot = 0
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                  tot += prob*(self.mdp.getReward(state,action,nextState)+self.discount*self.values[nextState])
                maxVal = max(tot,maxVal)
                temp[state] = maxVal
          self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        total = 0
        for nextState,prob in self.mdp.getTransitionStatesAndProbs(state,action):
          total += prob*(self.mdp.getReward(state,action,nextState)+self.discount*self.values[nextState])
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        if self.mdp.isTerminal(state):
          return None
        value = float("-inf")
        policy = None
        for action in self.mdp.getPossibleActions(state):
          temp = self.computeQValueFromValues(state,action)
          if temp>=value:
            value=temp
            policy=action
        return policy


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #for state in self.mdp.getStates():
        #  self.values[state] = 0

        #for iteration in range(0,self.iterations):
        #  state = self.mdp.getStates()[iteration%len(self.mdp.getStates())]
        #  if not self.mdp.isTerminal(state):
        #    self.values[state] = max(sum((self.discount*self.values[transState]+self.mdp.getReward(state))*prob for transState, prob in self.mdp.getTransitionStatesAndProbs(state,action)) for action in self.mdp.getPossibleActions(state))
        #temp = util.Counter()
        #for state in self.mdp.getStates():
        #  temp[state] = 0
        for iteration in range(0,self.iterations):
          #self.values = util.Counter()
          state = self.mdp.getStates()[iteration%len(self.mdp.getStates())]
          if self.mdp.isTerminal(state):
            self.values[state] = 0
          else:
          #if not self.mdp.isTerminal(state):
            maxVal = float("-inf") 
            for action in self.mdp.getPossibleActions(state):
              tot = 0
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                tot += prob*(self.mdp.getReward(state,action,nextState)+self.discount*self.values[nextState])
              maxVal = max(tot,maxVal)
            self.values[state] = maxVal


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateList = self.mdp.getStates()
        for state in self.mdp.getStates():
          self.values[state] = 0

        predecessors = {}
        for state in stateList:
          predecessors[state]=set()
        priorityState = util.PriorityQueue()
        for state in stateList:
          if not self.mdp.isTerminal(state):
            for action in self.mdp.getPossibleActions(state):
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                if prob>0:
                  predecessors[nextState].add(state)
            diff = abs(self.values[state] - max(self.getQValue(state,action) for action in self.mdp.getPossibleActions(state)))
            priorityState.push(state,-diff)
        for iteration in range(0,self.iterations):
          if priorityState.isEmpty():
            break
          state = priorityState.pop()
          self.values[state] = max(self.getQValue(state,action) for action in self.mdp.getPossibleActions(state))
          for pred in predecessors[state]:
            diff = abs(self.values[pred] - max(self.getQValue(pred,action) for action in self.mdp.getPossibleActions(pred)))
            if diff > self.theta:
              priorityState.update(pred,-diff)
