/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or find a way to pay it forward. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#ifndef __GHMM_H__
#define __GHMM_H__

#include <vector>

namespace GClasses {

class GHiddenMarkovModel
{
protected:
	int m_stateCount;
	int m_symbolCount;
	double* m_pInitialStateProbabilities;
	double* m_pTransitionProbabilities;
	double* m_pSymbolProbabilities;
	double* m_pTrainingBuffer;
	int m_maxLen;

public:
	GHiddenMarkovModel(int stateCount, int symbolCount);
	~GHiddenMarkovModel();

	/// Performs unit tests for this class. Throws an exception if there is a failure.
	static void test();

	/// Returns the current vector of initial state probabilities
	double* initialStateProbabilities() { return m_pInitialStateProbabilities; }

	/// Returns the current vector of transition probabilities, such that
	/// pTransitionProbabilities[stateCount * i + j] is the probability of
	/// transitioning from state i to state j.
	double* transitionProbabilities() { return m_pTransitionProbabilities; }

	/// Returns the current vector of symbol probabilities, such that
	/// pSymbolProbabilities[stateCount * i + j] is the probability of
	/// observing symbol j when in state i.
	double* symbolProbabilities() { return m_pSymbolProbabilities; }

	/// Calculates the log probability that the specified observation
	/// sequence would occur with this model.
	double forwardAlgorithm(const int* pObservations, int len);

	/// Finds the most likely state sequence to explain the specified
	/// observation sequence, and also returns the log probability of
	/// that state sequence given the observation sequence.
	double viterbi(int* pMostLikelyStates, const int* pObservations, int len);

	/// Uses expectation maximization to refine the model based on
	/// a training set of observation sequences. (You should have already
	/// set prior values for the initial, transition and symbol probabilites
	/// before you call this method.)
	void baumWelch(std::vector<int*>& sequences, std::vector<int>& lengths, int maxPasses = 0x7fffffff);

protected:
	void backwardAlgorithm(const int* pObservations, int len);
	void baumWelchBeginTraining(int maxLen);
	void baumWelchBeginPass();
	void baumWelchAddSequence(const int* pObservations, int len);
	double baumWelchEndPass();
	void baumWelchEndTraining();
};

} // namespace GClasses

#endif
