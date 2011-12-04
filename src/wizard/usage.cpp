#include "usage.h"
#include "../GClasses/GError.h"
#include "../GClasses/GString.h"
#include <cstring>

using namespace GClasses;
using std::string;
using std::vector;

UsageNode::UsageNode(const char* templ, const char* descr)
{
#ifdef DEBUG_HELPERS
	p0 = NULL; p1 = NULL; p2 = NULL; p3 = NULL;
#endif
	string s;
	while(true)
	{
		// skip whitespace
		while(*templ == ' ')
			templ++;
		if(*templ == '\0')
			break;

		// find the end of the name
		size_t i;
		for(i = 0; templ[i] != ' ' && templ[i] != '\0' && templ[i] != '='; i++)
		{
		}
		s.assign(templ, i);
		templ += i;
		m_parts.push_back(s);

		if(*templ == '=')
		{
			if(m_default_value.size() > 0)
				ThrowError("Only one default value is permitted per node. You should probably expand with child nodes.");

			// Find the end of the default value
			for(i = 1; templ[i] != ' ' && templ[i] != '\0'; i++)
			{
			}
			m_default_value.assign(templ + 1, i - 1);
			templ += i;
		}
	}
	m_description = descr;
#ifdef DEBUG_HELPERS
	if(m_parts.size() > 0) p0 = m_parts[0].c_str();
	if(m_parts.size() > 1) p1 = m_parts[1].c_str();
	if(m_parts.size() > 2) p2 = m_parts[2].c_str();
	if(m_parts.size() > 3) p3 = m_parts[3].c_str();
#endif
}

UsageNode::~UsageNode()
{
	for(vector<UsageNode*>::iterator it = m_choices.begin(); it != m_choices.end(); it++)
		delete(*it);
}

UsageNode* UsageNode::add(const char* templ, const char* descr)
{
	UsageNode* pNode = new UsageNode(templ, descr);
	m_choices.push_back(pNode);
	return pNode;
}

int UsageNode::findPart(const char* name)
{
	int index = 0;
	for(vector<string>::iterator it = m_parts.begin(); it != m_parts.end(); it++)
	{
		if(strcmp(it->c_str(), name) == 0)
			return index;
		index++;
	}
	return -1;
}

UsageNode* UsageNode::choice(const char* name)
{
	for(vector<UsageNode*>::iterator it = m_choices.begin(); it != m_choices.end(); it++)
	{
		if(std::strcmp((*it)->tok(), name) == 0)
			return *it;
	}
	return NULL;
}

void UsageNode::sig(string* pS)
{
	vector<string>::iterator it = m_parts.begin();
	pS->append(*it);
	for(it++; it != m_parts.end(); it++)
	{
		pS->append(" ");
		pS->append(*it);
	}
}

void UsageNode::print(std::ostream& stream, int depth, int tabSize, int maxWidth, int maxDepth, bool descriptions)
{
	// Print the token and args
	for(int i = 0; i < depth; i++)
		for(int j = 0; j < tabSize; j++)
			stream << " ";
	vector<string>::iterator it = m_parts.begin();
	stream << (*it);
	for(it++; it != m_parts.end(); it++)
	{
		stream << " ";
		stream << (*it);
	}
	stream << "\n";

	// Print the description
	if(descriptions)
	{
		GStringChopper sc(m_description.c_str(), 10, maxWidth - depth * tabSize, true);
		while(true)
		{
			const char* szLine = sc.next();
			if(!szLine)
				break;
			for(int i = 0; i <= depth; i++)
				for(int j = 0; j < tabSize; j++)
					stream << " ";
			stream << szLine;
			stream << "\n";
		}
	}

	// Print the children
	if(depth < maxDepth)
	{
		for(vector<UsageNode*>::iterator it = m_choices.begin(); it != m_choices.end(); it++)
			(*it)->print(stream, depth + 1, tabSize, maxWidth, maxDepth, descriptions);
	}
}

UsageNode* makeMasterUsageTree()
{
	UsageNode* pRoot = new UsageNode("[app]", "Welcome to the Waffles Wizard. This wizard will help you build a Waffles command.");
	pRoot->choices().push_back(makeAudioUsageTree());
	pRoot->choices().push_back(makeClusterUsageTree());
	pRoot->choices().push_back(makeDimRedUsageTree());
	pRoot->choices().push_back(makeGenerateUsageTree());
	pRoot->choices().push_back(makeLearnUsageTree());
	pRoot->choices().push_back(makePlotUsageTree());
	pRoot->choices().push_back(makeRecommendUsageTree());
	pRoot->choices().push_back(makeSparseUsageTree());
	pRoot->choices().push_back(makeTransformUsageTree());
	return pRoot;
};






UsageNode* makeAlgorithmUsageTree()
{
	UsageNode* pRoot = new UsageNode("[algorithm]", "A supervised learning algorithm, or a transductive algorithm.");
	{
		pRoot->add("agglomerativetransducer", "A model-free transduction algorithm based on single-link agglomerative clustering. Unlabeled patterns take the label of the cluster with which they are joined. It never joins clusters with different labels.");
	}
	{
		UsageNode* pBag = pRoot->add("bag <contents> end", "A bagging (bootstrap aggregating) ensemble. This is a way to combine the power of many learning algorithms through voting. \"end\" marks the end of the ensemble contents. Each algorithm instance is trained using a training set created by drawing (with replacement) from the original data until the training set has the same number of instances as the original data.");
		UsageNode* pContents = pBag->add("<contents>");
		pContents->add("[instance_count] [algorithm]", "Specify the number of instances of a learning algorithm to add to the bagging ensemble.");
	}
	{
		pRoot->add("baseline", "This is one of the simplest of all supervised algorithms. It ignores all features. For nominal labels, it always predicts the most common class in the training set. For continuous labels, it always predicts the mean label in the training set. An effective learning algorithm should never do worse than baseline--hence the name \"baseline\".");
	}
	{
		UsageNode* pBucket = pRoot->add("bucket <contents> end", "This uses cross-validation with the training set to select the best model from a bucket of models. When accuracy is measured across multiple datasets, it will usually do better than the best model in the bucket could do. \"end\" marks the end of the contents of the bucket.");
		UsageNode* pContents = pBucket->add("<contents>");
		pContents->add("[algorithm]", "Add an algorithm to the bucket");
	}
	{
		UsageNode* pBMA = pRoot->add("bma <contents> end", "A Bayesian model averaging ensemble. This trains each model after the manner of bagging, but then combines them weighted according to their probability given the data. Uniform priors are assumed.");
		UsageNode* pContents = pBMA->add("<contents>");
		pContents->add("[instance_count] [algorithm]", "Specify the number of instances of a learning algorithm to add to the BMA ensemble.");
	}
	{
		UsageNode* pBMC = pRoot->add("bmc <options> <contents> end", "A Bayesian model combination ensemble. This algorithm is described in Monteith, Kristine and Carroll, James and Seppi, Kevin and Martinez, Tony, Turning Bayesian Model Averaging into Bayesian Model Combination, Proceedings of the IEEE International Joint Conference on Neural Networks IJCNN'11, 2657--2663, 2011.");
		UsageNode* pOpts = pBMC->add("<options>");
		pOpts->add("-samples [n]=100", "Specify the number of samples to draw from the simplex of possible ensemble combinations. (Larger values result in better accuracy with the cost of more computation.)");
		UsageNode* pContents = pBMC->add("<contents>");
		pContents->add("[instance_count] [algorithm]", "Specify the number of instances of a learning algorithm to add to the BMA ensemble.");
	}
	{
		UsageNode* pBoost = pRoot->add("boost <options> [algorithm]", "Uses AdaBoost to create an ensemble that may be more accurate than a lone instance of the specified algorithm.");
		UsageNode* pOpts = pBoost->add("<options>");
		pOpts->add
		  ("-trainratio [value]=1.0", "When approximating the weighted training set by resampling, use a sample of size [value]*training_set_size");
		pOpts->add
		  ("-size [n]=30", "The number of base learners to use in "
		   "the ensemble.");
	}
	{
		pRoot->add("cvdt [n]=50", "This is a bucket of two bagging ensembles: one with [n] entropy-reducing decision trees, and one with [n] meanmarginstrees. (This algorithm is specified in Gashler, Michael S. and Giraud-Carrier, Christophe and Martinez, Tony. Decision Tree Ensemble: Small Heterogeneous Is Better Than Large Homogeneous. In The Seventh International Conference on Machine Learning and Applications, Pages 900 - 905, ICMLA '08. 2008)");
	}
	{
		UsageNode* pDT = pRoot->add("decisiontree <options>", "A decision tree.");
		UsageNode* pOpts = pDT->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-random [draws]=1", "Use random divisions (instead of divisions that reduce entropy). Random divisions make the algorithm train faster, and also increase model variance, so it is better suited for ensembles, but random divisions also make the decision tree more vulnerable to problems with irrelevant features. [draws] is typically 1, but if you specify a larger value, it will pick the best out of the specified number of random draws.");
		pOpts->add("-leafthresh [n]=1", "When building the tree, if the number of samples is <= this value, it will stop trying to divide the data and will create a leaf node. The default value is 1. For noisy data, larger values may be advantageous.");
		pOpts->add("-maxlevels [n]=5", "When building the tree, if the depth (the length of the path from the root to the node currently being formed, including the root and the currently forming node) is [n], it will stop trying to divide the data and will create a leaf node.  This means that there will be at most [n]-1 splits before a decision is made.  This crudely limits overfitting, and so can be helpful on small data sets.  It can also make the resulting trees easier to interpret.  If set to 0, then there is no maximum (which is the default).");
	}
	{
		UsageNode* pGCT = pRoot->add("graphcuttransducer <options>", "This is a model-free transduction algorithm. It uses a min-cut/max-flow graph-cut algorithm to separate each label from all of the others.");
		UsageNode* pOpts = pGCT->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-neighbors [k]=12", "Set the number of neighbors to connect with each point in order to form the graph.");
	}
	{
		UsageNode* pKNN = pRoot->add("knn <options>", "The k-Nearest-Neighbor instance-based learning algorithm. It uses Euclidean distance for continuous features and Hamming distance for nominal features.");
		UsageNode* pOpts = pKNN->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-neighbors [k]", "Specify the number of neighbors, k, to use.");
		pOpts->add("-equalweight", "Give equal weight to every neighbor. (The default is to use linear weighting for continuous features, and sqared linear weighting for nominal features.");
		pOpts->add("-scalefeatures", "Use a hill-climbing algorithm on the training set to scale the feature dimensions in order to give more accurate results. This increases training time, but also improves accuracy and robustness to irrelevant features.");
		pOpts->add("-pearson", "Use Pearson's correlation coefficient to evaluate the similarity between sparse vectors. (Only compatible with sparse training.)");
		pOpts->add("-cosine", "Use the cosine method to evaluate the similarity between sparse vectors. (Only compatible with sparse training.)");
	}
	{
		pRoot->add("linear", "A linear regression model");
	}
	{
		pRoot->add("meanmarginstree", "This is a very simple oblique (or linear combination) tree. (This algorithm is specified in Gashler, Michael S. and Giraud-Carrier, Christophe and Martinez, Tony. Decision Tree Ensemble: Small Heterogeneous Is Better Than Large Homogeneous. In The Seventh International Conference on Machine Learning and Applications, Pages 900 - 905, ICMLA '08. 2008)");
	}
	{
		UsageNode* pNB = pRoot->add("naivebayes <options>", "The naive Bayes learning algorithm.");
		UsageNode* pOpts = pNB->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-ess [value]=0.2", "Specifies an equivalent sample size to prevent unsampled values from dominating the joint distribution. Good values typically range between 0 and 1.5.");
	}
	{
		UsageNode* pNI = pRoot->add("naiveinstance <options>", "This is an instance learner that assumes each dimension is conditionally independant from other dimensions. It lacks the accuracy of knn in low dimensional feature space, but scales much better to high dimensionality.");
		UsageNode* pOpts = pNI->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-neighbors [k]=12", "Set the number of neighbors to use in each dimension");
	}
	{
		UsageNode* pNT = pRoot->add("neighbortransducer <options>", "This is a model-free transduction algorithm. It is an instance learner that propagates labels where the neighbors are most in agreement. This algorithm does well when classes sample a manifold (such as with text recognition).");
		UsageNode* pOpts = pNT->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters for this model with the current data.");
		pOpts->add("-neighbors [k]=12", "Set the number of neighbors to use with each point");
	}
	{
		UsageNode* pNN = pRoot->add("neuralnet <options>", "A single or multi-layer feed-forward neural network (a.k.a. multi-layer perceptron). It can be trained with online backpropagation (Rumelhart, D.E., Hinton, G.E., and Williams, R.J. Learning representations by back-propagating errors. Nature, 323:9, 1986.), or several other optimization methods.");
		UsageNode* pOpts = pNN->add("<options>");
		pOpts->add("-autotune", "Automatically determine a good set of parameters (including number of layers, hidden units, learning rate, etc.) for this model with the current data.");
		pOpts->add("-addlayer [size]=16", "Add a hidden layer with \"size\" logisitic units to the network. You may use this option multiple times to add multiple layers. The first layer added is adjacent to the input features. The last layer added is adjacent to the output labels. If you don't add any hidden layers, the network is just a single layer of sigmoid units.");
		pOpts->add("-learningrate [value]=0.1", "Specify a value for the learning rate. The default is 0.1");
		pOpts->add("-momentum [value]=0.0", "Specifies a value for the momentum. The default is 0.0");
		pOpts->add("-windowepochs [value]=200", "Specifies the number of training epochs that are performed before the stopping criteria is tested again. Bigger values will result in a more stable stopping criteria. Smaller values will check the stopping criteria more frequently.");
		pOpts->add("-minwindowimprovement [value]=0.002", "Specify the minimum improvement that must occur over the window of epochs for training to continue. [value] specifies the minimum decrease in error as a ratio. For example, if value is 0.02, then training will stop when the mean squared error does not decrease by two percent over the window of epochs. Smaller values will typically result in longer training times.");
		pOpts->add("-holdout [portion]=0.35", "Specify the portion of the data (between 0 and 1) to use as a hold-out set for validation. That is, this portion of the data will not be used for training, but will be used to determine when to stop training. If the holdout portion is set to 0, then no holdout set will be used, and the entire training set will be used for validation (which may lead to long training time and overfit).");
		pOpts->add("-dontsquashoutputs", "Don't squash the outputs values with the logistic function. Just report the net value at the output layer. This is often used for regression.");
		pOpts->add("-crossentropy", "Use cross-entropy instead of squared-error for the error signal.");
		UsageNode* pAct = pOpts->add("-activation [func]", "Specify the activation function to use with all subsequently added layers. (For example, if you add this option after all of the -addlayer options, then the specified activation function will only apply to the output layer. If you add this option before all of the -addlayer options, then the specified activation function will be used in all layers. It is okay to use a different activation function with each layer, if you want.)");
		{
			pAct->add("logistic", "The logistic sigmoid function. (This is the default activation function.)");
			pAct->add("arctan", "The arctan sigmoid function.");
			pAct->add("tanh", "The hyperbolic tangeant sigmoid function.");
			pAct->add("algebraic", "An algebraic sigmoid function.");
			pAct->add("identity", "The identity function. This activation function is used to create a layer of linear perceptrons. (For regression problems, it is common to use this activation function on the output layer.)");
			pAct->add("bidir", "A sigmoid-shaped function with a range from -inf to inf. It converges at both ends to -sqrt(-x) and sqrt(x). This activation function is designed to be used on the output layer with regression problems intead of identity.");
			pAct->add("gaussian", "A gaussian activation function");
			pAct->add("sinc", "A sinc wavelet activation function");
		}
	}
	{
		UsageNode* pRF = pRoot->add("randomforest [trees] <options>", "A baggging ensemble of decision trees that use random division boundaries. (This algorithm is described in Breiman, Leo (2001). Random Forests. Machine Learning 45 (1): 5-32. doi:10.1023/A:1010933404324.)");
		pRF->add("[trees]=50", "Specify the number of trees in the random forest");
		UsageNode* pOpts = pRF->add("<options>");
		pOpts->add("-samples [n]=1", "Specify the number of randomly-drawn attributes to evaluate. The one that maximizes information gain will be chosen for the decision boundary. If [n] is 1, then the divisions are completely random. Larger values will decrease the randomness.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}

	return pRoot;
}







UsageNode* makeAudioUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_audio [command]", "Tools for processing audio files.");
	{
		UsageNode* pAmplify = pRoot->add("amplify [in] [scalar] [out]", "Amplifies [in] by multiplying every sample by [scalar], then saves to [out].");
		pAmplify->add("[in]=in.wav", "The filename of an audio track in wav format.");
		pAmplify->add("[scalar]=1.0", "The amount to amplify. (Positive values less than 1 will reduce the volume. Values greater than 1 will increase the volume.)");
		pAmplify->add("[out]=out.wav", "The filename to which to save the results.");
	}
	{
		UsageNode* pMS = pRoot->add("makesilence [secs] [filename]", "Makes a wave file containing [secs] seconds of silence");
		pMS->add("[secs]=0.5", "The number of seconds of silence to generate.");
		pMS->add("[filename]=out.wav", "The output filename.");
		UsageNode* pOpts = pMS->add("<options>");
		pOpts->add("-bitspersample [n]=16", "Specify the number of bits-per-sample.");
		pOpts->add("-channels [n]=1", "Specify the number of channels.");
		pOpts->add("-samplerate [n]=44100", "Specify the number of samples-per-second.");
	}
	{
		UsageNode* pMS = pRoot->add("makesine [secs] [hertz] [filename]", "Makes a wave file containing [secs] seconds of a perfect sine wave with a frequency of [hertz].");
		pMS->add("[secs]=0.5", "The number of seconds of sound to generate.");
		pMS->add("[hertz]=440", "The frequency of the sound to generate.");
		pMS->add("[filename]=out.wav", "The output filename.");
		UsageNode* pOpts = pMS->add("<options>");
		pOpts->add("-bitspersample [n]=16", "Specify the number of bits-per-sample.");
		pOpts->add("-volumn [v]=1.0", "Specify the volume (from 0 to 1).");
		pOpts->add("-samplerate [n]=44100", "Specify the number of samples-per-second.");
	}
	{
		UsageNode* pMix = pRoot->add("mix [in1] [scalar1] [in2] [scalar2] [out]", "Mixes [in1] and [in2] to produce [out].");
		pMix->add("[in1]=in1.wav", "The filename of an audio track in wav format.");
		pMix->add("[scalar1]=1.0", "The amount to amplify [in1]. (Positive values less than 1 will reduce the volume. Values greater than 1 will increase the volume.)");
		pMix->add("[in2]=in2.wav", "The filename of an audio track in wav format.");
		pMix->add("[scalar2]=1.0", "The amount to amplify [in2]. (Positive values less than 1 will reduce the volume. Values greater than 1 will increase the volume.)");
		pMix->add("[out]=out.wav", "The filename to which to save the results.");
	}
	{
		UsageNode* pPS = pRoot->add("pitchshift [in] [halfsteps] [out]", "Shifts the pitch of a wav file (without changing the rate).");
		pPS->add("[in]=in.wav", "The filename of an audio track in wav format to pitch-shift.");
		pPS->add("[halfsteps]=12.0", "The number of half-steps to shift the audio track. Positive values will shift to a higher pitch. Negative values will shift to a lower pitch.");
		pPS->add("[out]=out.wav", "The filename to which to save the results.");
		UsageNode* pOpts = pPS->add("<options>");
		pOpts->add("-blocksize [n]=2048", "Specify the block size. [n] must be a power of 2.");
	}
	{
		UsageNode* pRedNoise = pRoot->add("reduceambientnoise [noise] [in] [out] <options>", "Learns the distribution in [noise], and subtracts it from [in] to generate [out].");
		pRedNoise->add("[noise]=noise.wav", "The filename of an audio track that samples ambient noise. (This is a recording of your studio with no deliberate noise--just computer fans, light hum, a/c, mic noise, or other nearly-constant sources of noise that are difficult to regulate.)");
		pRedNoise->add("[in]=in.wav", "The filename of an audio track in wav format from which you would like to remove the ambient noise.");
		pRedNoise->add("[out]=out.wav", "The filename to which to save the results.");
		UsageNode* pOpts = pRedNoise->add("<options>");
		pOpts->add("-blocksize [n]=2048", "Specify the size of the blocks in which the audio is processed. [n] must be a power of 2.");
		pOpts->add("-deviations [d]=2.5", "Specify the number of standard deviations from the noise mean to count as noise. Larger values will make it more aggressive at reducing noise, with more potential to disrupt the signal. Smaller (or negative) values can be used to make it less aggressive.");
	}
	{
		UsageNode* pSan = pRoot->add("sanitize [in] [out] <options>", "Finds segments of quiet in the input wave file, and replaces them with complete silence.");
		pSan->add("[in]=in.wav", "The filename of an audio track in wav format.");
		pSan->add("[out]=out.wav", "The filename to which to save the results.");
		UsageNode* pOpts = pSan->add("<options>");
		pOpts->add("-seconds [s]=0.1", "The length of time in seconds over which to decay the volume until it reaches zero. Only segments of at least twice this length will be sanitized (because there must be enough time to fade out, and then fade back in).");
		pOpts->add("-thresh [t]=0.15", "The volumne threshold (from 0 to 1). Segments that stay below this threshold for a sufficient length of time will be replaced with silence.");
	}
	{
		UsageNode* pSpec = pRoot->add("spectral [in] <options>", "Plots a spectral histogram of the frequencies in a wav file.");
		pSpec->add("[in]=in.wav", "The filename of an audio track in wav format.");
		UsageNode* pOpts = pSpec->add("<options>");
		pOpts->add("-start [pos]=0", "Specify the starting position in the wav file (in samples) to begin sampling.");
		pOpts->add("-size [n]=4096", "Specify the number of samples to take. This will also be the width of the resulting histogram in pixels. [n] must be a power of 2.");
		pOpts->add("-height [h]=512", "Specify the height of the chart in pixels.");
		pOpts->add("-out [filename]=plot.png", "Specify the output filename of the PNG image that is generated.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}









UsageNode* makeClusterUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_cluster [command]", "Cluster data.");
	pRoot->add("agglomerative [dataset] [clusters]", "Performs single-link agglomerative clustering. Outputs the cluster id for each row.");
	{
		UsageNode* pFKM = pRoot->add("fuzzykmeans [dataset] [clusters]", "Performs fuzzy k-means clustering. Outputs the cluster id for each row. This algorithm is specified in Li, D. and Deogun, J. and Spaulding, W. and Shuart, B., Towards missing data imputation: A study of fuzzy K-means clustering method, In Rough Sets and Current Trends in Computing, Springer, pages 573--579, 2004.");
		pFKM->add("[dataset]=in.arff", "The filename of a dataset to cluster.");
		UsageNode* pOpts = pFKM->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reps [n]=1", "Cluster the data [n] times, and return the clustering that minimizes the sum-weighted-distance between rows and the centroids.");
		pOpts->add("-fuzzifier [value]=1.3", "Specify the fuzzifier parameter, which should be greater than 1.");
	}
	{
		UsageNode* pKM = pRoot->add("kmeans [dataset] [clusters]", "Performs k-means clustering. Outputs the cluster id for each row.");
		pKM->add("[dataset]=in.arff", "The filename of a dataset to cluster.");
		UsageNode* pOpts = pKM->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reps [n]=1", "Cluster the data [n] times, and return the clustering that minimizes the sum-squared-distance between each row and its corresponding centroid.");
	}
	pRoot->add("kmedoids [dataset] [clusters]", "Performs k-medoids clustering. Outputs the cluster id for each row.");
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}






UsageNode* makeCollaborativeFilterUsageTree()
{
	UsageNode* pRoot = new UsageNode("[collab-filter]", "A collaborative-filtering recommendation algorithm.");
	{
		UsageNode* pBag = pRoot->add("bag <contents> end", "A bagging (bootstrap aggregating) ensemble. This is a way to combine the power of collaborative filtering algorithms through voting. \"end\" marks the end of the ensemble contents. Each collaborative filtering algorithm instance is trained on a subset of the original data, where each expressed element is given a probability of 0.5 of occurring in the training set.");
		UsageNode* pContents = pBag->add("<contents>");
		pContents->add("[instance_count] [collab-filter]", "Specify the number of instances of a collaborative filtering algorithm to add to the bagging ensemble.");
	}
	pRoot->add("baseline", "A very simple recommendation algorithm. It always predicts the average rating for each item. This algorithm is useful as a baseline algorithm for comparison.");
	{
		UsageNode* pClustDense = pRoot->add("clusterdense [n] <options>", "A collaborative-filtering algorithm that clusters users based on a dense distance metric with k-means, and then makes uniform recommendations within each cluster.");
		pClustDense->add("[n]=8", "The number of clusters to use.");
		UsageNode* pOpts = pClustDense->add("<options>");
		pOpts->add("-norm [l]=2.0", "Specify the norm for the L-norm distance metric to use.");
		pOpts->add("-missingpenalty [d]=1.0", "Specify the difference to use in the distance computation when a value is missing from one or both of the vectors.");
	}
	{
		UsageNode* pClustSparse = pRoot->add("clustersparse [n] <options>", "A collaborative-filtering algorithm that clusters users based on a sparse similarity metric with k-means, and then makes uniform recommendations within each cluster.");
		pClustSparse->add("[n]=8", "The number of clusters to use.");
		UsageNode* pOpts = pClustSparse->add("<options>");
		pOpts->add("-pearson", "Use Pearson Correlation to compute the similarity between users. (The default is to use the cosine method.)");
	}
	{
		UsageNode* pInst = pRoot->add("instance [k] <options>", "An instance-based collaborative-filtering algorithm that makes recommendations based on the k-nearest neighbors of a user.");
		pInst->add("[k]=256", "The number of neighbors to use.");
		UsageNode* pOpts = pInst->add("<options>");
		pOpts->add("-pearson", "Use Pearson Correlation to compute the similarity between users. (The default is to use the cosine method.)");
		pOpts->add("-regularize [value]=0.5", "Add [value] to the denominator in order to regularize the results. This ensures that recommendations will not be dominated when a small number of overlapping items occurs. Typically, [value] will be a small number, like 0.5 or 1.5.");
	}
	{
		UsageNode* pMF = pRoot->add("matrix [intrinsic] <options>", "A matrix factorization collaborative-filtering algorithm. (Implemented according to the specification on page 631 in Takacs, G., Pilaszy, I., Nemeth, B., and Tikk, D. Scalable collaborative filtering approaches for large recommender systems. The Journal of Machine Learning Research, 10:623-656, 2009. ISSN 1532-4435., except with the addition of learning-rate decay and a different stopping criteria.)");
		pMF->add("[intrinsic]=2", "The number of intrinsic (or latent) feature dims to use to represent each user's preferences.");
		UsageNode* pOpts = pMF->add("<options>");
		pOpts->add("-regularize [value]=0.0001", "Specify a regularization value. Typically, this is a small value. Larger values will put more pressure on the system to use small values in the matrix factors.");
	}
	{
		UsageNode* pNLPCA = pRoot->add("nlpca [intrinsic] <options>", "A non-linear PCA collaborative-filtering algorithm. This algorithm was published in Scholz, M. Kaplan, F. Guy, C. L. Kopka, J. Selbig, J., Non-linear PCA: a missing data approach, In Bioinformatics, Vol. 21, Number 20, pp. 3887-3895, Oxford University Press, 2005. It uses a generalization of backpropagation to train a multi-layer perceptron to fit to the known ratings, and to predict unknown values.");
		pNLPCA->add("[intrinsic]=2", "The number of intrinsic (or latent) feature dims to use to represent each user's preferences.");
		UsageNode* pOpts = pNLPCA->add("<options>");
		pOpts->add("-addlayer [size]=8", "Add a hidden layer with \"size\" logisitic units to the network. You may use this option multiple times to add multiple layers. The first layer added is adjacent to the input features. The last layer added is adjacent to the output labels. If you don't add any hidden layers, the network is just a single layer of sigmoid units.");
		pOpts->add("-learningrate [value]=0.1", "Specify a value for the learning rate. The default is 0.1");
		pOpts->add("-momentum [value]=0.0", "Specifies a value for the momentum. The default is 0.0");
		pOpts->add("-windowepochs [value]=10", "Specifies the number of training epochs that are performed before the stopping criteria is tested again. Bigger values will result in a more stable stopping criteria. Smaller values will check the stopping criteria more frequently.");
		pOpts->add("-minwindowimprovement [value]=0.0001", "Specify the minimum improvement that must occur over the window of epochs for training to continue. [value] specifies the minimum decrease in error as a ratio. For example, if value is 0.02, then training will stop when the mean squared error does not decrease by two percent over the window of epochs. Smaller values will typically result in longer training times.");
		pOpts->add("-dontsquashoutputs", "Don't squash the outputs values with the logistic function. Just report the net value at the output layer. This is often used for regression.");
		pOpts->add("-crossentropy", "Use cross-entropy instead of squared-error for the error signal.");
		pOpts->add("-noinputbias", "Do not use an input bias.");
		pOpts->add("-nothreepass", "Use one-pass training instead of three-pass training.");
		UsageNode* pAct = pOpts->add("-activation [func]", "Specify the activation function to use with all subsequently added layers. (For example, if you add this option after all of the -addlayer options, then the specified activation function will only apply to the output layer. If you add this option before all of the -addlayer options, then the specified activation function will be used in all layers. It is okay to use a different activation function with each layer, if you want.)");
		{
			pAct->add("logistic", "The logistic sigmoid function. (This is the default activation function.)");
			pAct->add("arctan", "The arctan sigmoid function.");
			pAct->add("tanh", "The hyperbolic tangeant sigmoid function.");
			pAct->add("algebraic", "An algebraic sigmoid function.");
			pAct->add("identity", "The identity function. This activation function is used to create a layer of linear perceptrons. (For regression problems, it is common to use this activation function on the output layer.)");
			pAct->add("bidir", "A sigmoid-shaped function with a range from -inf to inf. It converges at both ends to -sqrt(-x) and sqrt(x). This activation function is designed to be used on the output layer with regression problems intead of identity.");
			pAct->add("gaussian", "A gaussian activation function");
			pAct->add("sinc", "A sinc wavelet activation function");
		}
	}
	return pRoot;
}








UsageNode* makeDimRedUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_dimred [command]", "Reduce dimensionality, attribute selection, operations related to manifold learning, NLDR, etc.");
	{
		UsageNode* pAS = pRoot->add("attributeselector [dataset] <data_opts> <options>", "Make a ranked list of attributes from most to least salient. The ranked list is printed to stdout. Attributes are zero-indexed.");
		pAS->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pDO = pAS->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		UsageNode* pOpts = pAS->add("<options>");
		pOpts->add("-out [n] [filename]", "Save a dataset containing only the [n]-most salient features to [filename].");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-labeldims [n]=1", "Specify the number of dimensions in the label (output) vector. The default is 1. (Don't confuse this with the number of class labels. It only takes one dimension to specify a class label, even if there are k possible labels.)");
	}
	{
		UsageNode* pBE = pRoot->add("blendembeddings [data-orig] [neighbor-finder] [data-a] [data-b] <options>", "Compute a blended \"average\" embedding from two reduced-dimensionality embeddings of some data.");
		pBE->add("[data-orig]=orig.arff", "The filename of the original high-dimensional data.");
		pBE->add("[data-a]=a.arff", "The first reduced dimensional embedding of [data-orig]");
		pBE->add("[data-b]=b.arff", "The second reduced dimensional embedding of [data-orig]");
		UsageNode* pOpts = pBE->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pBFU = pRoot->add("breadthfirstunfolding [dataset] [neighbor-finder] [target_dims] <options>", "A manifold learning algorithm.");
		UsageNode* pOpts = pBFU->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reps [n]=10", "The number of times to compute the embedding and blend the results together. If not specified, the default is 1.");
		pBFU->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pBFU->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
	}
	{
		UsageNode* pIsomap = pRoot->add("isomap [dataset] [neighbor-finder] [target_dims] <options>", "Use the Isomap algorithm to reduce dimensionality.");
		UsageNode* pOpts = pIsomap->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-tolerant", "If there are points that are disconnected from the rest of the graph, just drop them from the data. (This may cause the results to contain fewer rows than the input.)");
		pIsomap->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pIsomap->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
	}
	UsageNode* pSOM = pRoot->add
	  ("som [dataset] [dimensions] <options>",
	   "Give the output of a Kohonen self-organizing map with the given "
           "dimensions trained on the input dataset.  Ex: \"som foo 10 11\" "
	   "would train a 10x11 map on the input data and then give its "
	   "2D output for each of the input points as a row in the output "
	   "file.");
	{
	  pSOM->add("[dataset]","The filename of a .arff file to be "
		    "transformed.");
	  pSOM->add("[dimensions]","A list of integers, one for each "
		    "dimension of the map being created, giving the number "
		    "of nodes in that dimension.");
	  UsageNode* pOpts = pSOM->add("<options>");
	  {
	    pOpts->add("-tofile [filename]",
		       "Write the trained map to the given filename");
	    pOpts->add("-fromfile [filename]",
		       "Read a map from the file rather than training it");
	    pOpts->add("-seed [integer]",
		       "Seed the random number generator with integer to "
		       "obtain reproducible results");
	    pOpts->add("-neighborhood [gaussian|uniform]",
		       "Use the specified neighborhood type to determine the "
		       "influence of a node on its neighbors.");
	    pOpts->add("-printMeshEvery [numIter] [baseFilename] [xDim] [yDim] <showTrain>", "Print a 2D-Mesh visualization every numIter training iteratons to an svg file generated from baseFilename.  The x dimension and y dimension will be chosen from the zero-indexed dimensions of the input using xDim and yDim.  If the option \"showTrain\" is present then the training data is displayed along with the mesh. Ex. \"-printMeshEvery 2 foo 0 1 showTrain\" will write foo_01.svg foo_02.svg etc. every other iteration using the first two dimensions of the input and also display the training data in the svg image.  Note that including this option twice will create two different printing actions, allowing multiple dimension pairs to be visualized at once.");
	    pOpts->add("-batchTrain [startWidth] [endWidth] [numEpochs] [numConverge]",
		       "Trains the network using the batch training algorithm. "
		       "Neighborhood decreases exponentially from startWidth "
		       "to endWidth over numEpochs epochs.  Each epoch lasts "
		       "at most numConverge passes through the dataset "
		       "waiting for the network to converge.  Do not ignore "
		       "numConverge=1.  There has been good performance with "
		       "this on some datasets.  This is the default training "
		       "algorithm.");
	    pOpts->add("-stdTrain [startWidth] [endWidth] [startRate] [endRate] [numIter]",
		       "Trains the network using the standard incremental "
		       "training algorithm with the network width decreasing "
		       "exponentially from startWidth to endWidth and the "
		       "learning rate also decreasing exponentially from "
		       "startRate to endRate, this will happen in exactly "
		       "numIter data point presentations.");
	  }
	}
	
	{
		UsageNode* pLLE = pRoot->add("lle [dataset] [neighbor-finder] [target_dims] <options>", "Use the LLE algorithm to reduce dimensionality.");
		UsageNode* pOpts = pLLE->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pLLE->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pLLE->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
	}
	{
		UsageNode* pMS = pRoot->add("manifoldsculpting [dataset] [neighbor-finder] [target_dims] <options>", "Use the Manifold Sculpting algorithm to reduce dimensionality. (This algorithm is specified in Gashler, Michael S. and Ventura, Dan and Martinez, Tony. Iterative non-linear dimensionality reduction with manifold sculpting. In Advances in Neural Information Processing Systems 20, pages 513-520, MIT Press, Cambridge, MA, 2008.)");
		pMS->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pMS->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
		UsageNode* pOpts = pMS->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-continue [dataset]=prev.arff", "Continue refining the specified reduced-dimensional results. (This feature enables Manifold Sculpting to improve upon its own results, or to refine the results from another dimensionality reduction algorithm.)");
		pOpts->add("-scalerate [value]=0.9999", "Specify the scaling rate. If not specified, the default is 0.999. A value close to 1 will give better results, but will cause the algorithm to take longer.");
	}
	{
		UsageNode* pMDS = pRoot->add("multidimensionalscaling [distance-matrix] [target-dims]", "Perform MDS on the specified [distance-matrix].");
		pMDS->add("[distance-matrix]=distances.arff", "The filename of an arff file that contains the pair-wise distances (or dissimilarities) between every pair of points. It must be a square matrix of real values. Only the upper-triangle of this matrix is actually used. The lower-triangle and diagonal is ignored.");
		UsageNode* pOpts = pMDS->add("<options>");
		pOpts->add("-squareddistances", "The distances in the distance matrix are squared distances, instead of just distances.");
	}
	{
		UsageNode* pNeuroPCA = pRoot->add("neuropca [dataset] [target_dims] <options>", "Projects the data into the specified number of dimensions with a non-linear generalization of principle component analysis. (Prints results to stdout. The input file is not modified.)");
		UsageNode* pOpts = pNeuroPCA->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-clampbias", "Do not let the bias drift from the centroid. (Leaving the bias unclamped typically gives better results with non-linear activation functions. Clamping them to the centroid is necessary if you want results equivalent with PCA.)");
		pOpts->add("-linear", "Use a linear activation function instead of the default logistic activation function. (The logistic activation function typically gives better results with most problems, but the linear activation function may be used to obtain results equivalent to PCA.)");
		pNeuroPCA->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pNeuroPCA->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
	}
	{
		UsageNode* pPCA = pRoot->add("pca [dataset] [target_dims] <options>", "Projects the data into the specified number of dimensions with principle component analysis. (Prints results to stdout. The input file is not modified.)");
		UsageNode* pOpts = pPCA->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-roundtrip [filename]=roundtrip.arff", "Do a lossy round-trip of the data and save the results to the specified file.");
		pOpts->add("-eigenvalues [filename]=eigenvalues.arff", "Save the eigenvalues to the specified file.");
		pOpts->add("-components [filename]=eigenvectors.arff", "Save the centroid and principal component vectors (in order of decreasing corresponding eigenvalue) to the specified file.");
		pOpts->add("-aboutorigin", "Compute the principal components about the origin. (The default is to compute them relative to the centroid.)");
		pPCA->add("[dataset]=in.arff", "The filename of the high-dimensional data to reduce.");
		pPCA->add("[target_dims]=2", "The number of dimensions to reduce the data into.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}








UsageNode* makeGenerateUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_generate [command]", "Generate certain useful datasets");
	{
		UsageNode* pCrane = pRoot->add("crane <options>", "Generate a dataset where each row represents a ray-traced image of a crane with a ball.");
		UsageNode* pOpts = pCrane->add("<options>");
		pOpts->add("-saveimage [filename]=frames.png", "Save an image showing all the frames.");
		pOpts->add("-ballradius [size]=0.3", "Specify the size of the ball. The default is 0.3.");
		pOpts->add("-frames [horiz] [vert]", "Specify the number of frames to render.");
		pOpts->add("-size [wid] [hgt]", "Specify the size of each frame.");
		pOpts->add("-blur [radius]=5.0", "Blurs the images. A good starting value might be 5.0.");
		pOpts->add("-gray", "Use a single grayscale value for every pixel instead of three (red, green, blue) channel values.");
	}
	pRoot->add("cube [n]=2000", "returns data evenly distributed on the surface of a unit cube. Each side is sampled with [n]x[n] points. The total number of points in the dataset will be 6*[n]*[n]-12*[n]+8.");
	{
		UsageNode* pES = pRoot->add("entwinedspirals [points] <options>", "Generates points that lie on an entwined spirals manifold.");
		pES->add("[points]=1000", "The number of points with which to sample the manifold.");
		UsageNode* pOpts = pES->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reduced", "Generate intrinsic values instead of extrinsic values. (This might be useful to empirically measure the accuracy of a manifold learner.)");
	}
	{
		UsageNode* pFishBowl = pRoot->add("fishbowl [n] <option>", "Generate samples on the surface of a fish-bowl manifold.");
		pFishBowl->add("[n]=2000", "The number of samples to draw.");
		UsageNode* pOpts = pFishBowl->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-opening [size]=0.25", "the size of the opening. (0.0 = no opening. 0.25 = default. 1.0 = half of the sphere.)");
	}
	{
		UsageNode* pGRW = pRoot->add("gridrandomwalk [arff-file] [width] [samples] <options>", "Generate a sequence of action-observation pairs by randomly walking around on a grid of observation vectors. Assumes there are four possible actions consisting of up, down, left, right.");
		pGRW->add("[arff-file]=grid.arff", "The filename of an arff file containing observation vectors arranged in a grid.");
		pGRW->add("[width]=20", "The width of the grid.");
		pGRW->add("[samples]=4000", "The number of samples to take. In other words, the length of the random walk.");
		UsageNode* pOpts = pGRW->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-start [x] [y]", "Specifies the starting state. The default is to start in the center of the grid.");
		pOpts->add("-obsfile [filename]=observations.arff", "Specify the filename for the observation sequence data. The default is observations.arff.");
		pOpts->add("-actionfile [filename]=actions.arff", "Specify the filename for the actions data. The default is actions.arff.");
	}
	{
		UsageNode* pITON = pRoot->add("imagetranslatedovernoise [png-file] <options>", "Sample a manifold by translating an image over a background of noise.");
		pITON->add("[png-file]=in.png", "The filename of a png image.");
		UsageNode* pOpts = pITON->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reduced", "Generate intrinsic values instead of extrinsic values. (This might be useful to empirically measure the accuracy of a manifold learner.)");
	}
	{
		UsageNode* pManifold = pRoot->add("manifold [samples] <options> [equations]", "Generate sample points randomly distributed on the surface of a manifold.");
		pManifold->add("[samples]=2000", "The number of points with which to sample the manifold");
		UsageNode* pOpts = pManifold->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pManifold->add("[equations]=\"y1(x1,x2)=x1;y2(x1,x2)=sqrt(x1*x2);h(x)=sqrt(1-x);y3(x1,x2)=x2*x2-h(x1)\"", "A set of equations that define the manifold. The equations that define the manifold must be named y1, y2, ..., but helper equations may be included. The manifold-defining equations must all have the same number of parameters. The parameters will be drawn from a standard normal distribution (from 0 to 1). Usually it is a good idea to wrap the equations in quotes. Example: \"y1(x1,x2)=x1;y2(x1,x2)=sqrt(x1*x2);h(x)=sqrt(1-x);y3(x1,x2)=x2*x2-h(x1)\"");
	}
	{
		UsageNode* pNoise = pRoot->add("noise [rows] <options>", "Generate random data by sampling from a distribution.");
		pNoise->add("[rows]=1000", "The number of patterns to generate.");
		UsageNode* pOpts = pNoise->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		UsageNode* pDist = pOpts->add("-dist [distribution]", "Specify the distribution. The default is normal 0 1");
		pDist->add("beta [alpha] [beta]");
		pDist->add("binomial [n] [p]");
		pDist->add("categorical 3 [p0] [p1] [p2]", "A categorical distribution with 3 classes. [p0], [p1], and [p2] specify the probabilities of each of the 3 classes. (This is just an example. Other values besides 3 may be used for the number of classes.)");
		pDist->add("cauchy [median] [scale]");
		pDist->add("chisquare [t]");
		pDist->add("exponential [beta]");
		pDist->add("f [t] [u]");
		pDist->add("gamma [alpha] [beta]");
		pDist->add("gaussian [mean] [deviation]");
		pDist->add("geometric [p]");
		pDist->add("logistic [mu] [s]");
		pDist->add("lognormal [mu] [sigma]");
		pDist->add("normal [mean] [deviation]");
		pDist->add("poisson [mu]");
		pDist->add("softimpulse [s]");
		pDist->add("spherical [dims] [radius]");
		pDist->add("student [t]");
		pDist->add("uniform [a] [b]");
		pDist->add("weibull [gamma]");
	}
	{
		UsageNode* pRS = pRoot->add("randomsequence [length] <options>", "Generates a sequential list of integer values, shuffles them randomly, and then prints the shuffled list to stdout.");
		pRS->add("[length]=10", "The number of values in the random sequence.");
		UsageNode* pOpts = pRS->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-start [value]=0", "Specify the smallest value in the sequence.");
	}
	{
		UsageNode* pScalRot = pRoot->add("scalerotate [png-file] <options>", "Generate a dataset where each row represents an image that has been scaled and rotated by various amounts. Thus, these images form an open-cylinder (although somewhat cone-shaped) manifold.");
		pScalRot->add("[png-file]=image.png", "The filename of a PNG image");
		UsageNode* pOpts = pScalRot->add("<options>");
		pOpts->add("-saveimage [filename]=frames.png", "Save a composite image showing all the frames in a grid.");
		pOpts->add("-frames [rotate-frames] [scale-frames]", "Specify the number of frames. The default is 40 15.");
		pOpts->add("-arc [radians]=1.570796", "Specify the rotation amount. If not specified, the default is 6.2831853... (2*PI).");
	}
	{
		UsageNode* pSCurve = pRoot->add("scurve [points] <options>", "Generate points that lie on an s-curve manifold.");
		pSCurve->add("[points]=2000", "The number of points with which to sample the manifold");
		UsageNode* pOpts = pSCurve->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reduced", "Generate intrinsic values instead of extrinsic values. (This might be useful to empirically measure the accuracy of a manifold learner.)");
	}
	{
		UsageNode* pSIR = pRoot->add("selfintersectingribbon [points] <options>", "Generate points that lie on a self-intersecting ribbon manifold.");
		pSIR->add("[points]=2000", "The number of points with which to sample the manifold.");
		UsageNode* pOpts = pSIR->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pSR = pRoot->add("swissroll [points] <options>", "Generate points that lie on a swiss roll manifold.");
		pSR->add("[points]=2000", "The number of points with which to sample the manifold.");
		UsageNode* pOpts = pSR->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-reduced", "Generate intrinsic values instead of extrinsic values. (This might be useful to empirically measure the accuracy of a manifold learner.)");
		pOpts->add("-cutoutstar", "Don't sample within a star-shaped region on the manifold.");
	}
	{
		UsageNode* pWI = pRoot->add("windowedimage [png-file] <options>", "Sample a manifold by translating a window over an image. Each pattern represents the windowed portion of the image.");
		pWI->add("[png-file]=in.png", "The filename of the png image from which to generate the data.");
		UsageNode* pOpts = pWI->add("<options>");
		pOpts->add("-reduced", "Generate intrinsic values instead of extrinsic values. (This might be useful to empirically measure the accuracy of a manifold learner.)");
		pOpts->add("-stepsizes [horiz] [vert]", "Specify the horizontal and vertical step sizes. (how many pixels to move the window between samples.)");
		pOpts->add("-windowsize [width] [height]", "Specify the size of the window. The default is half the width and height of [png-file].");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}

	return pRoot;
}








UsageNode* makeLearnUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_learn [command]", "Supervised learning, transduction, cross-validation, etc.");
	{
		UsageNode* pAT = pRoot->add("autotune [dataset] <data_opts> [algname]", "Use cross-validation to automatically determine a good set of parameters for the specified algorithm with the specified data. The selected parameters are printed to stdout.");
		pAT->add("[dataset]=train.arff", "The filename of a dataset.");
		UsageNode* pDO = pAT->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		UsageNode* pAlgs = pAT->add("[algname]", "The name of the algorithm that you wish to automatically tune.");
		pAlgs->add("agglomerativetransducer", "An agglomerative transducer");
		pAlgs->add("decisiontree", "A decision tree");
		pAlgs->add("graphcuttransducer", "A graph-cut transducer");
		pAlgs->add("knn", "A k-nearest-neighbor instance-based learner");
		pAlgs->add("meanmarginstree", "A mean margins tree");
		pAlgs->add("neuralnet", "A feed-foward neural network (a.k.a. multi-layer perceptron)");
		pAlgs->add("naivebayes", "A naive Bayes model");
		pAlgs->add("naiveinstance", "A naive instance model");
	}
	{
		UsageNode* pTrain = pRoot->add("train <options> [dataset] <data_opts> [algorithm]", "Trains a supervised learning algorithm. The trained model-file is printed to stdout. (Typically, you will want to pipe this to a file.)");
		UsageNode* pOpts = pTrain->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-calibrate", "Calibrate the model after it is trained, such that predicted distributions will approximate the distributions represented in the training data. This switch is typically used only if you plan to predict distributions (by calling predictdistribution) instead of just class labels or regression values. Calibration will not effect the predictions made by regular calls to 'predict', which is used by most other tools.");
		pTrain->add("[dataset]=train.arff", "The filename of a dataset.");
		UsageNode* pDO = pTrain->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pPredict = pRoot->add("predict <options> [model-file] [dataset] <data_opts>", "Predict labels for all of the patterns in [dataset]. Results are printed in the form of a \".arff\" file (including both features and predictions) to stdout.");
		UsageNode* pOpts = pPredict->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pPredict->add("[model-file]=model.json", "The filename of a trained model. (This is the file to which you saved the output when you trained a supervised learning algorithm.)");
		pPredict->add("[dataset]=test.arff", "The filename of a dataset. (There should already be placeholder labels in this dataset. The placeholder labels will be replaced in the output by the labels that the model predicts.)");
		UsageNode* pDO = pPredict->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pPOP = pRoot->add("predictdistribution <options> [model-file] [data-set] <data_opts> [pattern]", "Predict a distribution for a single feature vector and print it to stdout. (Typically, the '-calibrate' switch should be used when training the model. If the model is not calibrated, then the predicted distribution may not be a very good estimated distribution. Also, some models cannot be used to predict a distribution.)");
		UsageNode* pOpts = pPOP->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pPOP->add("[model-file]=model.json", "The filename of a trained model. (This is the file to which you saved the output when you trained a supervised learning algorithm.)");
		pPOP->add("[data-set]=train.arff", "The filename of dataset from which to obtain meta-data. This can be the training set or the test set. It doesn't matter which, because the data is ignored. Only the meta-data, such as the string names of attribute values, are obtained from this dataset.");
		pPOP->add("[pattern]", "A list of feature values separated by spaces. (A \"?\" may be used for unknown feature values if the model supports using unknown feature values.)");
		UsageNode* pDO = pPOP->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pTest = pRoot->add("test <options> [model-file] [dataset] <data_opts>", "Test a trained model using some test data. Results are printed to stdout for each dimension in the label vector. Predictive accuracy is reported for nominal label dimensions, and mean-squared-error is reported for continuous label dimensions.");
		UsageNode* pOpts = pTest->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-confusion", "Print a confusion matrix for each nominal label attribute.");
		pTest->add("[model-file]=model.json", "The filename of a trained model. (This is the file to which you saved the output when you trained a supervised learning algorithm.)");
		pTest->add("[dataset]=test.arff", "The filename of a test dataset. (This dataset must have the same number of columns as the dataset with which the model was trained.)");
		UsageNode* pDO = pTest->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pTransduce = pRoot->add("transduce <options> [labeled-set] <data_opts1> [unlabeled-set] <data_opts2> [algorithm]", "Predict labels for [unlabeled-set] based on the examples in [labeled-set]. For most algorithms, this is the same as training on [labeled-set] and then predicting labels for [unlabeled-set]. Some algorithms, however, have no models. These can transduce, even though they cannot be trained. The predicted labels are printed to stdout as a \".arff\" file.");
		UsageNode* pOpts = pTransduce->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pTransduce->add("[labeled-set]=train.arff", "The filename of a dataset. The labels in this dataset are used to infer labels for the unlabeled set.");
		pTransduce->add("[unlabeled-set]=test.arff", "The filename of a dataset. This dataset must have placeholder labels, but these will be ignored when predicting new labels.");
		UsageNode* pDO1 = pTransduce->add("<data_opts1>");
		pDO1->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO1->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		UsageNode* pDO2 = pTransduce->add("<data_opts2>");
		pDO2->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO2->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pTransAcc = pRoot->add("transacc <options> [training-set] <data_opts1> [test-set] <data_opts2> [algorithm]", "Measure the transductive accuracy of [algorithm] with respect to the specified training and test sets. Results are printed to stdout for each dimension in the label vector. Predictive accuracy is reported for nominal labels, and mean-squared-error is reported for continuous labels.");
		UsageNode* pOpts = pTransAcc->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-confusion", "Print a confusion matrix for each nominal label attribute.");
		pTransAcc->add("[training-set]=train.arff", "The filename of a dataset. The labels in this dataset are used to infer labels for the unlabeled set.");
		pTransAcc->add("[test-set]=test.arff", "The filename of a dataset. This dataset must have placeholder labels. The placeholder labels will be replaced in the output with the new predicted labels.");
		UsageNode* pDO1 = pTransAcc->add("<data_opts1>");
		pDO1->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO1->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		UsageNode* pDO2 = pTransAcc->add("<data_opts2>");
		pDO2->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO2->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pSplitTest = pRoot->add("splittest <options> [dataset] <data_opts> [algorithm]", "This shuffles the data, then splits it into two parts, trains with one part, and tests with the other. (This also works with model-free algorithms.) Results are printed to stdout for each dimension in the label vector. Predictive accuracy is reported for nominal labels, and mean-squared-error is reported for continuous labels.");
		UsageNode* pOpts = pSplitTest->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-trainratio [value]=0.5", "Specify the amount of the data (between 0 and 1) to use for training. The rest will be used for testing.");
		pOpts->add("-reps [value]=10", "Specify the number of repetitions to perform. If not specified, the default is 1.");
		pOpts->add("-writelastmodel [filename]", 
			   "Write the model generated on the last repetion "
			   "to the given filename.  Note that this only works "
			   "when the learner being used has an internal "
			   "model.");
		pOpts->add("-confusion", "Print a confusion matrix for each nominal label attribute after each repetition.");
		pOpts->add("-stddev", "Print the standard deviation of the results at the end as well as their mean.");
		pSplitTest->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pDO = pSplitTest->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pCV = pRoot->add("crossvalidate <options> [dataset] <data_opts> [algorithm]", "Perform cross-validation with the specified dataset and algorithm. Results are printed to stdout. (Supports model-free algorithms too.)");
		UsageNode* pOpts = pCV->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-reps [value]=5", "Specify the number of repetitions to perform. If not specified, the default is 5.");
		pOpts->add("-folds [value]=2", "Specify the number of folds to use. If not specified, the default is 2.");
		pOpts->add("-succinct", "Just report the average accuracy. Do not report deviation, or results at each fold.");
		pCV->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pDO = pCV->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pPR = pRoot->add("precisionrecall <options> [dataset] <data_opts> [algorithm]", "Compute the precision/recall for a dataset and algorithm");
		UsageNode* pOpts = pPR->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-labeldims [n]=1", "Specify the number of dimensions in the label (output) vector. The default is 1. (Don't confuse this with the number of class labels. It only takes one dimension to specify a class label, even if there are k possible labels.)");
		pOpts->add("-reps [n]=5", "Specify the number of reps to perform. More reps means it will take longer, but results will be more accurate. The default is 5.");
		pOpts->add("-samples [n]=100", "Specify the granularity at which to measure recall. If not specified, the default is 100.");
		pPR->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pDO = pPR->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pSter = pRoot->add("sterilize <options> [dataset] <data_opts> [algorithm]", "Perform cross-validation to generate a new dataset that contains only the correctly-classified instances. The new sterilized data is printed to stdout.");
		UsageNode* pOpts = pSter->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-folds [n]=10", "Specify the number of cross-validation folds to perform.");
		pOpts->add("-diffthresh [d]=0.1", "Specify a threshold of absolute difference for continuous labels. Predictions with an absolute difference less than this threshold are considered to be \"correct\".");
		pSter->add("[dataset]=data.arff", "The filename of a dataset to sterilize.");
		UsageNode* pDO = pSter->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pTR = pRoot->add("trainrecurrent <options> [method] [obs-data] [action-data] [context-dims] [algorithm] [algorithm]", "Train a recurrent model of a dynamical system with the specified training [method]. The training data is specified by [obs-data], which specifies the sequence of observations, and [action-data], which specifies the sequence of actions. [context-dims] specifies the number of dimensions in the state-space of the system. The two algorithms specify the two functions of a model of a dynamical system. The first [algorithm] models the transition function. The second [algorithm] models the observation function.");
		UsageNode* pOpts = pTR->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pOpts->add("-paramdims 2 [wid] [hgt]", "If observations are images, use this option to parameterize the predictions, so only the channel values of each pixel are predicted. (Other values besides 2 dimensions are also supported.)");
		pOpts->add("-state [filename]=state.arff", "Save the estimated state to the specified file. (Only has effect if moses is used as the training method.)");
		pOpts->add("-validate [interval] 1 [obs] [action]", "Perform validation at [interval]-second intervals with observation data, [obs], and action data, [action]. (Also supports more than 1 validation sequence if desired.)");
		pOpts->add("-out [filename]=model.json", "Save the resulting model to the specified file. If not speicified, the default is \"model.json\".");
		pOpts->add("-noblur", "Do not use blurring. The default is to use blurring. Sometimes blurring improves results. Sometimes not.");
		pOpts->add("-traintime [seconds]=3600", "Specify how many seconds to train the model. The default is 3600, which is 1 hour.");
		pOpts->add("-isomap", "Use Isomap instead of Breadth-first Unfolding if moses is used as the training method.");

		UsageNode* pMeth = pTR->add("[method]");
		pMeth->add("moses", "Use Temporal-NLDR to estimate state, then build the model using the state estimate.");
		pMeth->add("bptt [depth] [iters-per-grow-sequence]", "Backpropagation Through Time. [depth] specifies the number of instances of the transition function that will appear in the unfolded model. A good value might be 3. [iters-per-grow-sequence] specifies the number of pattern presentations before the sequence is incremented. A good value might be 50000.");
		pMeth->add("evolutionary", "Train with evoluationary optimization");
		pMeth->add("hillclimber", "Train with a hill-climbing algorithm.");
		pMeth->add("annealing [deviation] [decay] [window]", "Train with simulated annealing. Good values might be 2.0 0.5 300");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}







UsageNode* makeNeighborUsageTree()
{
	UsageNode* pRoot = new UsageNode("[neighbor-finder]", "A neighbor-finding algorithm.");
	{
		UsageNode* pBF = pRoot->add("bruteforce <options> [k]", "The slow way to find the neareast Euclidean-distance neighbors.");
		UsageNode* pOpts = pBF->add("<options>");
		UsageNode* pCC = pOpts->add("-cyclecut [thresh]", "Use CycleCut to break shortcuts and cycles.");
		pCC->add("[thresh]=10", "The threshold cycle-length for bad cycles.");
		pOpts->add("-normalize", "Normalize distances in local neighborhoods so that all neighborhoood have a uniform amount of total distance.");
		pBF->add("[k]=12", "The number of neighbors.");
	}
	{
		UsageNode* pKD = pRoot->add("kdtree <options> [k]", "A faster way to find the neareast Euclidean-distance neighbors.");
		UsageNode* pOpts = pKD->add("<options>");
		UsageNode* pCC = pOpts->add("-cyclecut [thresh]", "Use CycleCut to break shortcuts and cycles.");
		pCC->add("[thresh]=10", "The threshold cycle-length for bad cycles.");
		pKD->add("[k]=12", "The number of neighbors.");
	}
	{
		UsageNode* pMan = pRoot->add("saffron <options> [cands] [k] [t] [thresh]", "The SAFFRON intelligent neighbor-finding algorithm that finds neighborhoods with aligned tangent hyperplanes. This algorithm was published in Gashler, Michael S. and Martinez, Tony. Tangent space guided intelligent neighbor finding. In Proceedings of the IEEE International Joint Conference on Neural Networks IJCNN’11, pages 2617–2624, IEEE Press, 2011.");
		UsageNode* pOpts = pMan->add("<options>");
		UsageNode* pCC = pOpts->add("-cyclecut [thresh]", "Use CycleCut to break shortcuts and cycles.");
		pCC->add("[thresh]=10", "The threshold cycle-length for bad cycles.");
		pMan->add("[cands]=32", "The median number of neighbors to use as candidates.");
		pMan->add("[k]=8", "The number of neighbors to find for each point.");
		pMan->add("[t]=2", "The number of dimensions in the tangent hyperplanes.");
		pMan->add("[thresh]=0.9", "A threshold above which all sqared-correlation values are considered to be equal.");
	}
	{
		UsageNode* pSys = pRoot->add("temporal <options> [action-data] [k]", "A neighbor-finder designed for use in modeling certain types of dynamical systems. It estimates the number of time-steps between observations. This algorithm was published in Gashler, Michael S. and Martinez, Tony. Temporal nonlinear dimensionality reduction. In Proceedings of the IEEE International Joint Conference on Neural Networks IJCNN’11, pages 1959–1966, IEEE Press, 2011.");
		UsageNode* pOpts = pSys->add("<options>");
		UsageNode* pCC = pOpts->add("-cyclecut [thresh]", "Use CycleCut to break shortcuts and cycles.");
		pCC->add("[thresh]=10", "The threshold cycle-length for bad cycles.");
		pSys->add("[action-data]=actions.arff", "The filename of an arff file for the sequence of actions given to the system.");
		pSys->add("[k]=12", "The number of neighbors.");
	}
	return pRoot;
}








UsageNode* makePlotUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_plot [command]", "Visualize data, plot functions, make charts, etc.");
	{
		UsageNode* p3d = pRoot->add("3d [dataset] <options>", "Make a 3d scatter plot. Points are colored with a spectrum according to their order in the dataset.");
		p3d->add("[dataset]=data.arff", "The filename of a dataset to plot. It must have exactly 3 continuous attributes.");
		UsageNode* pOpts = p3d->add("<options>");
		pOpts->add("-blast", "Produce a 5-by-5 grid of renderings, each time using a random point of view. It will print the random camera directions that it selects to stdout.");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-size [width] [height]", "Sets the size of the image. The default is 1000 1000.");
		pOpts->add("-pointradius [radius]=40.0", "Set the size of the points. The default is 40.0.");
		pOpts->add("-bgcolor [color]=ddeeff", "Set the background color. If not specified, the default is ffffff.");
		pOpts->add("-cameradistance [dist]=3.5", "Set the distance between the camera and the mean of the data. This value is specified as a factor, which is multiplied by the distance between the min and max corners of the data. If not specified, the default is 1.5. (If the camera is too close to the data, make this value bigger.)");
		pOpts->add("-cameradirection [dx] [dy] [dz]", "Specifies the direction from the camera to the mean of the data. (The camera always looks at the mean.) The default is 0.6 -0.3 -0.8.");
		pOpts->add("-out [filename]=plot.png", "Specify the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-nolabels", "Don't put axis labels on the bounding box.");
		pOpts->add("-nobox", "Don't draw a bounding box around the plot.");
	}
	{
		UsageNode* pBar = pRoot->add("bar [dataset] <options>", "Make a bar chart.");
		pBar->add("[dataset]=data.arff", "The filename of a dataset for the bar chart. The dataset must contain exactly one continuous attribute. Each data row specifies the height of a bar.");
		UsageNode* pOpts = pBar->add("<options>");
		pOpts->add("-log", "Use a logarithmic scale.");
		pOpts->add("-out [filename]=plot.png", "Specifies the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
	}
	pRoot->add("bigo [dataset]=results.arff", "Estimate the Big-O runtime of algorithms based on empirical results. Regresses the formula t=a*(n^b+c) to fit the data, where n is the value in attribute 0 (representing the size of the data), and t (representing time) in the other attributes for each algorithm. The values of a, b, and c are reported for each attribute > 0.");
	{
		UsageNode* pEquat = pRoot->add("equation <options> [equations]", "Plot an equation (or multiple equations) in 2D");
		UsageNode* pOpts = pEquat->add("<options>");
		pOpts->add("-out [filename]=plot.png", "Specify the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-size [width] [height]", "Specify the size of the chart. (The default is 1024 1024.)");
		pOpts->add("-range [xmin] [ymin] [xmax] [ymax]", "Set the range. (The default is: -10 -10 10 10.)");
		pOpts->add("-textsize [size]=1.0", "Sets the label font size. If not specified, the default is 2.0.");
		pOpts->add("-nogrid", "Do not draw any grid lines.");
		pEquat->add("[equations]=\"f1(x)=sin(x)/x\"", "A set of equations separated by semicolons. Since '^' is a special character for many shells, it's usually a good idea to put your equations inside quotation marks. Here are some examples:\n"
		"\"f1(x)=3*x+2\"\n"
		"\"f1(x)=(g(x)+1)/g(x); g(x)=sqrt(x)+pi\"\n"
		"\"h(bob)=bob^2;f1(x)=3+bar(x,5)*h(x)-(x/foo);bar(a,b)=a*b-b;foo=3.2\"\n"
		"Only functions that begin with 'f' followed by a number will be plotted, starting with 'f1', and it will stop when the next number in ascending order is not defined. You may define any number of helper functions or constants with any name you like. Built in constants include: e, and pi. Built in functions include: +, -, *, /, %, ^, abs, acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, erf, floor, gamma, lgamma, log, max, min, sin, sinh, sqrt, tan, and tanh. These generally have the same meaning as in C, except '^' means exponent, \"gamma\" is the gamma function, and max and min can support any number (>=1) of parameters. (Some of these functions may not not be available on Windows, but most of them are.) You can override any built in constants or functions with your own variables or functions, so you don't need to worry too much about name collisions. Variables must begin with an alphabet character or an underscore. Multiplication is never implicit, so you must use a '*' character to multiply. Whitespace is ignored.");
	}
	{
		UsageNode* pHist = pRoot->add("histogram [dataset] <options>", "Make a histogram.");
		pHist->add("[dataset]=samples.arff", "The filename of a dataset for the histogram.");
		UsageNode* pOpts = pHist->add("<options>");
		pOpts->add("-size [width] [height]", "Specify the size of the chart. (The default is 1024 1024.)");
		pOpts->add("-attr [index]=0", "Specify which attribute is charted. (The default is 0.)");
		pOpts->add("-out [filename]=hist.png", "Specify the name of the output file. (If not specified, the default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-range [xmin] [xmax] [ymax]", "Specify the range of the histogram plot");
	}
	{
		UsageNode* pModel = pRoot->add("model [model-file] [dataset] [attr-x] [attr-y] <options>", "Plot the model space of a trained supervised learning algorithm.");
		pModel->add("[model-file]=model.json", "The filename of the trained model. (You can use \"waffles_learn train\" to make a model file.)");
		pModel->add("[dataset]=train.arff", "The filename of a dataset to be plotted. It can be the training set that was used to train the model, or a test set that it hasn't yet seen.");
		pModel->add("[attr-x]=0", "The zero-based index of a continuous feature attributes for the horizontal axis.");
		pModel->add("[attr-y]=1", "The zero-based index of a continuous feature attributes for the vertical axis.");
		UsageNode* pOpts = pModel->add("<options>");
		pOpts->add("-out [filename]=plot.png", "Specify the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-size [width] [height]", "Specify the size of the image.");
		pOpts->add("-pointradius [size]=3.0", "Specify the size of the dots used to represent each instance.");
	}
	{
		UsageNode* pOL = pRoot->add("overlay [png1] [png2] <options>", "Make an image comprised of [png1] with [png2] on top of it. The two images must be the same size.");
		pOL->add("[png1]=below.png", "The filename of an image in png format.");
		pOL->add("[png2]=above.png", "The filename of an image in png format.");
		UsageNode* pOpts = pOL->add("<options>");
		pOpts->add("-out [filename]=plot.png", "Specify the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-backcolor [hex]=00ff00", "Specify the six-digit hexadecimal representation of the background color. (This color will be treated as being transparent in [png2]. If not specified, the default is ffffff (white).");
		pOpts->add("-tolerance [n]=12", "Specify the tolerance (an integer). If not specified, the default is 0. If a larger value is specified, then pixels in [png2] that are close to the background color will also be treated as being transparent.");
	}
	{
		UsageNode* pOver = pRoot->add("overview [dataset]", "Generate a matrix of plots of attribute distributions and correlations. This is a useful chart for becoming acquainted with a dataset.");
		pOver->add("[dataset]=data.arff", "The filename of a dataset to be charted.");
		UsageNode* pOpts = pOver->add("<options>");
		pOpts->add("-out [filename]=plot.png", "Specify the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-cellsize [value]=100", "Change the size of each cell. The default is 100.");
		pOpts->add("-jitter [value]=0.03", "Specify how much to jitter the plotted points. The default is 0.03.");
		pOpts->add("-maxattrs [value]=20", "Specifies the maximum number of attributes to plot. The default is 20.");
	}
	{
		UsageNode* pPDT = pRoot->add("printdecisiontree [model-file] <dataset> <data_opts>", "Print a textual representation of a decision tree to stdout.");
		pPDT->add("[model-file]=model.json", "The filename of a trained decision tree model. (You can make one with the command \"waffles_learn train [dataset] decisiontree > [filename]\".)");
		pPDT->add("<dataset>", "An optional filename of the arff file that was used to train the decision tree. The data in this file is ignored, but the meta-data will be used to make the printed model richer.");
		UsageNode* pDO = pPDT->add("<data_opts>");
		pDO->add("-labels [attr_list]=0", "Specify which attributes to use as labels. (If not specified, the default is to use the last attribute for the label.) [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
		pDO->add("-ignore [attr_list]=0", "Specify attributes to ignore. [attr_list] is a comma-separated list of zero-indexed columns. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	{
		UsageNode* pScat = pRoot->add("scatter [dataset] <options>", "Makes a scatter plot or line graph.");
		pScat->add("[dataset]=data.arff", "The filename of a dataset to be plotted. The first attribute specifies the values on the horizontal axis. All other attributes specify the values on the vertical axis for a certain color.");
		UsageNode* pOpts = pScat->add("<options>");
		pOpts->add("-lines", "Draw lines connecting sequential point in the data. (In other words, make a line graph instead of a scatter plot.)");
		pOpts->add("-size [width] [height]", "Specify the size of the chart. (The default is 1024 1024.)");
		pOpts->add("-logx", "Show the horizontal axis on a logarithmic scale");
		pOpts->add("-logy", "Show the vertical axis on a logarithmic scale");
		pOpts->add("-nogrid", "Do not draw any grid lines. (This is the same as doing both -novgrid and -nohgrid.)");
		pOpts->add("-novgrid", "Do not draw any vertical grid lines.");
		pOpts->add("-nohgrid", "Do not draw any horizontal grid lines.");
		pOpts->add("-textsize [size]=1.0", "Sets the label font size. If not specified, the default is 2.0.");
		pOpts->add("-pointradius [radius]=7.0", "Set the size of the point dots. If not specified, the default is 7.0.");
		pOpts->add("-linethickness [value]=3.0", "Specify the line thickness. (The default is 3.0.)");
		pOpts->add("-range [xmin] [ymin] [xmax] [ymax]", "Sets the range. (The default is to determine the range automatically.)");
		pOpts->add("-aspect", "Adjust the range to preserve the aspect ratio. In other words, make sure that both axes visually have the same scale.");
		pOpts->add("-chartcolors [background] [text] [grid]", "Sets colors for the specified areas. (The default is ffffff 000000 808080.)");
		pOpts->add("-linecolors [c1] [c2] [c3] [c4]", "Sets the colors for the first four attributes. The default is 0000a0 a00000 008000 504010 (blue, red, green, brown). (If there are more than four lines, it will just distribute them evenly over the color spectrum.)");
		pOpts->add("-spectrum", "Instead of giving each line a unique color, this will use the color spectrum to indicate the position of each point within the data.");
		pOpts->add("-specmod [cycle]=20", "Like -spectrum, except it repeats the spectrum with the specified cycle size.");
		pOpts->add("-out [filename]=plot.png", "Specifies the name of the output file. (The default is plot.png.) It should have the .png extension because other image formats are not yet supported.");
		pOpts->add("-neighbors [neighbor-finder]", "Draw lines connecting each point with its neighbors as determined by the specified neighbor finding algorithm.");
	}
	pRoot->add("percentsame [dataset1] [dataset2]", "Given two "
		   "data files, counts the number of identical values in the "
		   "same place in each dataset.  Prints as a percent for "
		   "each column.  The data files must have the same "
		   "number and type of attributes as well as the same number "
		   "of rows.");
	{
	UsageNode* pSemMap = pRoot->add
	  ("semanticmap [model-file] [dataset] <options>",
	   "Write a svg file representing a semantic map for the given "
	   "self-organizing map processing the given dataset.  For each node "
	   "n, a semantic map plots, at "
	   "n's location in the map, one attribute (usually a class label) of "
	   "the entry of the input data to which n responds most strongly.");
	{
	  pSemMap->add("[model-file]","The self-organizing map output "
		       "from \"waffles_transform som\".");
	  pSemMap->add("[dataset]","Data for the semantic map in .arff "
		       "format.  Any attributes over the number needed for "
		       "input to the self-organizing map are ignored in "
		       "determining som node responses.");
	  UsageNode* pOpts = pSemMap->add("<options>");
	  {
	    pOpts->add("-out [filename]", "Write the svg file to filename.  "
		       "The default is \"semantic_map.svg\".");
	    pOpts->add("-labels [column]","Use the attribute column for "
		       "labeling.  Column is a zero-based index into the "
		       "attributes.  The default is to use the last column.");
	    pOpts->add("-variance","Label the nodes with the variance of the "
		       "label column values for their winning dataset entries. "
		       "If the label column is a variable being predicted, "
		       "then its variance its related to the predictive power "
		       "of that node.  Higher variance means lower predictive "
		       "power.");
	  }
	}
		UsageNode* pStats = pRoot->add("stats [dataset]", "Prints some basic stats about the dataset to stdout.");
		pStats->add("[dataset]=data.arff", "The filename of a dataset.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}








UsageNode* makeRecommendUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_recommend [command]", "Predict missing values in data, and test collaborative-filtering recommendation systems.");
	{
		UsageNode* pCV = pRoot->add("crossvalidate <options> [3col-data] [collab-filter]", "Measure accuracy using cross-validation. Prints MSE and MAE to stdout.");
		UsageNode* pOpts = pCV->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-folds [n]=2", "Specify the number of folds. If not specified, the default is 2.");
		pCV->add("[3col-data]=ratings.arff", "The filename of 3-column dataset with one row for each rating. Column 0 contains a user ID. Column 1 contains an item ID. Column 2 contains the known rating for that user-item pair.");
	}
	{
		UsageNode* pFMV = pRoot->add("fillmissingvalues <options> [data] [collab-filter]", "Fill in the missing values in an ARFF file with predicted values and print the resulting full dataset to stdout.");
		UsageNode* pOpts = pFMV->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pFMV->add("[data]=data.arff", "The filename of a dataset with missing values to impute.");
	}
	{
		UsageNode* pPR = pRoot->add("precisionrecall <options> [3col-data] [collab-filter]", "Compute precision-recall data");
		UsageNode* pOpts = pPR->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-ideal", "Ignore the model and compute ideal results (as if the model always predicted correct ratings).");
		pPR->add("[3col-data]=ratings.arff", "The filename of 3-column dataset with one row for each rating. Column 0 contains a user ID. Column 1 contains an item ID. Column 2 contains the known rating for that user-item pair.");
	}
	{
		UsageNode* pROC = pRoot->add("roc <options> [3col-data] [collab-filter]", "Compute data for an ROC curve. (The area under the curve will appear in the comments at the top of the data.)");
		UsageNode* pOpts = pROC->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-ideal", "Ignore the model and compute ideal results (as if the model always predicted correct ratings).");
		pROC->add("[3col-data]=ratings.arff", "The filename of 3-column dataset with one row for each rating. Column 0 contains a user ID. Column 1 contains an item ID. Column 2 contains the known rating for that user-item pair.");
	}
	{
		UsageNode* pTransacc = pRoot->add("transacc <options> [train] [test] [collab-filter]", "Train using [train], then test using [test]. Prints MSE and MAE to stdout.");
		UsageNode* pOpts = pTransacc->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pTransacc->add("[train]=train.arff", "The filename of 3-column dataset with one row for each rating. Column 0 contains a user ID. Column 1 contains an item ID. Column 2 contains the known rating for that user-item pair.");
		pTransacc->add("[test]=test.arff", "The filename of 3-column dataset with one row for each rating. Column 0 contains a user ID. Column 1 contains an item ID. Column 2 contains the known rating for that user-item pair.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}











UsageNode* makeSparseUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_sparse [command]", "Sparse learning, document classification, information retrieval, etc.");
	{
		UsageNode* pDocsToSparse = pRoot->add("docstosparsematrix <options> [folder1] [folder2] ...", "Converts a set of documents to a sparse feature matrix, and a dense label matrix. [folder1] should contain all of the documents in class1. [folder2] should contain all the documents in class2, and so forth. The words are filtered against a common set of stop words. Also, words less than 4 letters are ignored. Currently, only .txt and .html documents are supported. Other file types are ignored. Each row in the sparse matrix represents one of the documents. Subdirectories are not followed. The feature vector is saved to a sparse matrix in compressed-column format. If more than one folder is specified, then a dense label matrix will also be generated. A mapping from row number to document filename is printed to stdout.");
		UsageNode* pOpts = pDocsToSparse->add("<options>");
		pOpts->add("-nostem", "Specifies not to stem the words. (The default is to use the Porter stemming algorithm.)");
		pOpts->add("-binary", "Just use the value 1 if the word occurs in a document, or a 0 if it does not occur. The default behavior is to compute the somewhat more meaningful value: a/b*log(c/d), where a=the number of times the word occurs in this document, b=the max number of times this word occurs in any document, c=total number of documents, and d=number of documents that contain this word.");
		pOpts->add("-out [features-filename] [labels-filename]", "Specify the filenames for the sparse feature matrix and the dense labels matrix. Note that if only one folder of documents is provided, then [labels-filename] will be ignored (since all documents come from the same folder/class), but a bogus filename must be provided for it anyway.");
		pOpts->add("-vocabfile [filename]=vocab.txt", "Save the vocabulary of words to the specified file. The default is to not save the list of words. Note that the words will be stemmed (unless -nostem was specified), so it is normal for many of them to appear misspelled.");
	}
	{
		UsageNode* pTrain = pRoot->add("train <options> [sparse-features] [dense-labels] [algorithm]", "Train the specified algorithm with the sparse matrix. Only incremental learners (such as naivebayes or neuralnet) support this functionality. It will print the trained model-file to stdout.");
		UsageNode* pOpts = pTrain->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pTrain->add("[sparse-features]=features.sparse", "The filename of a sparse matrix representing the training features. (This matrix should not contain labels.)");
		pTrain->add("[dense-labels]=labels.arff", "The filename of a dense matrix representing the training labels that correspond with the training features. (The label matrix must have the same number of rows as the feature matrix.)");
	}
	{
		UsageNode* pPredict = pRoot->add("predict <options> [model-file] [sparse-matrix]", "Predict labels for all of the rows in [sparse-matrix]. Label predictions for each row are printed to stdout. (The features are not printed with the predictions.)");
		UsageNode* pOpts = pPredict->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pPredict->add("[sparse-matrix]=features.sparse", "The filename of a sparse matrix of features for which labels should be predicted. (The feature matrix should not contain labels.)");
		pPredict->add("[model-file]=model.json", "The filename of a trained model. (This is the file to which you saved the output when you trained a supervised learning algorithm.) Only incremental learning algorithms are supported.");
	}
	{
		UsageNode* pShuffle = pRoot->add("shuffle [sparse-matrix] <options>", "Shuffles the row order of a sparse matrix.");
		pShuffle->add("[sparse-matrix]=features.arff", "The filename of a sparse matrix.");
		UsageNode* pOpts = pShuffle->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pSplit = pRoot->add("split [sparse-matrix] [rows] [filename1] [filename2]", "Splits a sparse matrix into two datasets. Nothing is printed to stdout.");
		pSplit->add("[sparse-matrix]=features.arff", "The filename of a sparse matrix.");
		pSplit->add("[rows]=200", "The number of rows to put in the first file. The rest go in the second file.");
	}
	{
		UsageNode* pSplitFold = pRoot->add("splitfold [dataset] [i] [n] <options>", "Divides a sparse dataset into [n] parts of approximately equal size, then puts part [i] into one file, and the other [n]-1 parts in another file. (This tool may be useful, for example, to implement n-fold cross validation.)");
		pSplitFold->add("[dataset]=features.arff", "The filename of a sparse datset.");
		pSplitFold->add("[i]=0", "The (zero-based) index of the fold, or the part to put into the training set. [i] must be less than [n].");
		pSplitFold->add("[n]=10", "The number of folds.");
		UsageNode* pOpts = pSplitFold->add("<options>");
		pOpts->add("-out [train_filename] [test_filename]", "Specify the filenames for the training and test portions of the data. The default values are train.sparse and test.sparse.");
	}
	{
		UsageNode* pTest = pRoot->add("test <options> [model-file] [sparse-features] [labels]", "Evaluates predictive accuracy for nominal labels, and mean-squared-error for continuous labels. Prints the score to stdout.");
		UsageNode* pOpts = pTest->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator. (Use this option to ensure that your results are reproduceable.)");
		pTest->add("[model-file]=model.json", "The filename of a trained model. (This is the file to which you saved the output when you trained a supervised learning algorithm.) Only incremental learning algorithms are supported.");
		pTest->add("[sparse-features]=features.sparse", "The filename of a sparse matrix of features for which labels should be predicted. (The feature matrix should not contain labels.)");
		pTest->add("[labels]=labels.arff", "The filename of a dense matrix of labels in .arff format. (This matrix should have the same number of rows as [sparse-features], since they correspond with each other.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}






UsageNode* makeTransformUsageTree()
{
	UsageNode* pRoot = new UsageNode("waffles_transform [command]", "Transform data, shuffle rows, swap columns, matrix operations, etc.");
	{
		UsageNode* pAdd = pRoot->add("add [dataset1] [dataset2]", "Adds two matrices together element-wise. Results are printed to stdout.");
		pAdd->add("[dataset1]=a.arff", "The filename of the first matrix.");
		pAdd->add("[dataset2]=b.arff", "The filename of the second matrix.");
	}
	{
		UsageNode* pAIC = pRoot->add("addindexcolumn [dataset] <options>", "Add a column that Specify the index of each row. This column will be inserted as column 0. (For example, suppose you would like to plot the values in each column of your data against the row index. Most plotting tools expect one of the columns to supply the position on the horizontal axis. This feature will create such a column for you.)");
		pAIC->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pOpts = pAIC->add("<options>");
		pOpts->add("-start [value]=0.0", "Specify the initial index. (the default is 0.0).");
		pOpts->add("-increment [value]=1.0", "Specify the increment amount. (the default is 1.0).");
	}
	{
		UsageNode* pAN = pRoot->add("addnoise [dataset] [dev] <options>", "Add Gaussian noise with the specified deviation to all the elements in the dataset. (Assumes that the values are all continuous.)");
		pAN->add("[dataset]=data.arff", "The filename of a dataset.");
		pAN->add("[dev]=1.0", "The deviation of the Gaussian noise");
		UsageNode* pOpts = pAN->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-excludelast [n]=1", "Do not add noise to the last [n] columns.");
	}
	{
		pRoot->add("aggregatecols [n]=0", "Make a matrix by aggregating each column [n] from the .arff files in the current directory. The resulting matrix is printed to stdout.");
	}
	{
		pRoot->add("aggregaterows [n]=0", "Make a matrix by aggregating each row [n] from the .arff files in the current directory. The resulting matrix is printed to stdout.");
	}
	{
		UsageNode* pAlign = pRoot->add("align [a] [b]", "Translates and rotates dataset [b] to minimize mean squared difference with dataset [a]. (Uses the Kabsch algorithm.)");
		pAlign->add("[a]=base.arff", "The filename of a dataset.");
		pAlign->add("[b]=alignme.arff", "The filename of a dataset.");
	}
	pRoot->add("autocorrelation [dataset]=data.arff", "Compute the autocorrelation of the specified time-series data.");
	pRoot->add("cholesky [dataset]=in.arff", "Compute the cholesky decomposition of the specified matrix.");
	{
		UsageNode* pCorr = pRoot->add("correlation [dataset] [attr1] [attr2] <options>", "Compute the linear correlation coefficient of the two specified attributes.");
		pCorr->add("[dataset]=data.arff", "The filename of a dataset.");
		pCorr->add("[attr1]=0", "A zero-indexed attribute number.");
		pCorr->add("[attr2]=1", "A zero-indexed attribute number.");
		UsageNode* pOpts = pCorr->add("<options>");
		pOpts->add("-aboutorigin", "Compute the correlation about the origin. (The default is to compute it about the mean.)");
	}
	{
		UsageNode* pCumCols = pRoot->add("cumulativecolumns [dataset] [column-list]", "Accumulates the values in the specified columns. For example, a column that contains the values 2,1,3,2 would be changed to 2,3,6,8. This might be useful for converting a histogram of some distribution into a histogram of the cumulative disribution.");
		pCumCols->add("[dataset]=data.arff", "The filename of a dataset.");
		pCumCols->add("[column-list]=0", "A comma-separated list of zero-indexed columns to transform. A hypen may be used to specify a range of columns. Example: 0,2-5,7");
	}
	{
		pRoot->add("determinant [dataset]=m.arff", "Compute the determinant of the specified matrix.");
	}
	{
		UsageNode* pDisc = pRoot->add("discretize [dataset] <options>", "Discretizes the continuous attributes in the specified dataset.");
		pDisc->add("[dataset]=in.arff", "The filename of a dataset.");
		UsageNode* pOpts = pDisc->add("<options>");
		pOpts->add("-buckets [count]=10", "Specify the number of buckets to use. If not specified, the default is to use the square root of the number of rows in the dataset.");
		pOpts->add("-colrange [first] [last]", "Specify a range of columns. Only continuous columns in the specified range will be modified. (Columns are zero-indexed.)");
	}
	{
		UsageNode* pDropCols = pRoot->add("dropcolumns [dataset] [column-list]", "Remove one or more columns from a dataset and prints the results to stdout. (The input file is not modified.)");
		pDropCols->add("[dataset]=in.arff", "The filename of a dataset.");
		pDropCols->add("[column-list]=0", "A comma-separated list of zero-indexed columns to drop. A hypen may be used to specify a range of columns.  A '*' preceding a value means to index from the right instead of the left. For example, \"0,2-5\" refers to columns 0, 2, 3, 4, and 5. \"*0\" refers to the last column. \"0-*1\" refers to all but the last column.");
	}
	pRoot->add("dropmissingvalues [dataset]=data.arff", "Remove all rows that contain missing values.");
	{
		UsageNode* pDRV = pRoot->add("droprandomvalues [dataset] [portion] <options>", "Drop random values from the specified dataset. The resulting dataset with missing values is printed to stdout.");
		pDRV->add("[dataset]=in.arff", "The filename of a dataset.");
		pDRV->add("[portion]=0.1", "The portion of the data to drop. For example, if [portion] is 0.1, then 10% of the values will be replaced with unknown values");
		UsageNode* pOpts = pDRV->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pEx = pRoot->add("export [dataset] <options>", "Print the data as a list of comma separated values without any meta-data.");
		pEx->add("[dataset]=data.arff", "The filename of a dataset.");
		UsageNode* pOpts = pEx->add("<options>");
		pOpts->add("-tab", "Separate with tabs instead of commas.");
		pOpts->add("-space", "Separate with spaces instead of commas.");
	}
	{
		UsageNode* pDR = pRoot->add("droprows [dataset] [after-size]", "Removes all rows except for the first [after-size] rows.");
		pDR->add("[dataset]=data.arff", "The filename of a dataset.");
		pDR->add("[after-size]=1", "The number of rows to keep");
	}
	{
		UsageNode* pFMS = pRoot->add("fillmissingvalues [dataset] <options>", "Replace all missing values in the dataset. (Note that the fillmissingvalues command in the waffles_recommend tool performs a similar task, but it can intelligently predict the missing values instead of just using the baseline value.)");
		pFMS->add("[dataset]=data.arff", "The filename of a dataset");
		UsageNode* pOpts = pFMS->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
		pOpts->add("-random", "Replace each missing value with a randomly chosen non-missing value from the same attribute. (The default is to use the baseline value. That is, the mean for continuous attributes, and the most-common value for nominal attributes.)");
	}
	{
		UsageNode* pIm = pRoot->add("import [dataset] <options>", "Convert a text file of comma separated (or otherwise separated) values to a .arff file. The meta-data is automatically determined. The .arff file is printed to stdout. This makes it easy to operate on structured data from a spreadsheet, database, or pretty-much any other source.");
		pIm->add("[dataset]=in.arff", "The filename of a dataset.");
		UsageNode* pOpts = pIm->add("<options>");
		pOpts->add("-tab", "Data elements are separated with a tab character instead of a comma.");
		pOpts->add("-space", "Data elements are separated with a single space instead of a comma.");
		pOpts->add("-whitespace", "Data elements are separated with an arbitrary amount of whitespace.");
		pOpts->add("-semicolon", "Data elements are separated with semicolons.");
		pOpts->add("-separator [char]='#'", "Data elements are separated with the specified character.");
		pOpts->add("-columnnames", "Use the first row of data for column names.");
	}
	{
		UsageNode* pEV = pRoot->add("enumeratevalues [dataset] [col]", "Enumerates all of the unique values in the specified column, and replaces each value with its enumeration. (For example, if you have a column that contains the social-security-number of each user, this will change them to numbers from 0 to n-1, where n is the number of unique users.)");
		pEV->add("[dataset]=data.arff", "The filename of a dataset");
		pEV->add("[col]=0", "The column index (starting with 0) to enumerate");
	}
	{
		UsageNode* pMSE = pRoot->add("measuremeansquarederror [dataset1] [dataset2] <options>", "Print the mean squared error between two datasets. (Both datasets must be the same size.)");
		pMSE->add("[dataset1]=a.arff", "The filename of a dataset");
		pMSE->add("[dataset2]=b.arff", "The filename of a dataset");
		UsageNode* pOpts = pMSE->add("<options>");
		pOpts->add("-fit", "Use a hill-climber to find an affine transformation to make dataset2 fit as closely as possible to dataset1. Report results after each iteration.");
		pOpts->add("-sum", "Sum the mean-squared error over each attribute and only report this sum. (The default is to report the mean-squared error in each attribute.)");
	}
	{
		UsageNode* pMH = pRoot->add("mergehoriz [dataset1] [dataset2]", "Merge two (or more) datasets horizontally. All datasets must already have the same number of rows. The resulting dataset will have all the columns of both datasets.");
		pMH->add("[dataset1]=a.arff", "The filename of a dataset");
		pMH->add("[dataset2]=b.arff", "The filename of a dataset");
	}
	{
		UsageNode* pNode = pRoot->add("mergevert [dataset1] [dataset2]", "Merge two datasets vertically. Both datasets must already have the same number of columns. The resulting dataset will have all the rows of both datasets.");
		pNode->add("[dataset1]=a.arff", "The filename of a dataset");
		pNode->add("[dataset2]=b.arff", "The filename of a dataset");
	}
	{
		UsageNode* pMult1 = pRoot->add("multiply [a] [b] <options>", "Matrix multiply [a] x [b]. Both arguments are the filenames of .arff files. Results are printed to stdout.");
		pMult1->add("[dataset1]=a.arff", "The filename of a dataset");
		pMult1->add("[dataset2]=b.arff", "The filename of a dataset");
		UsageNode* pOpts = pMult1->add("<options>");
		pOpts->add("-transposea", "Transpose [a] before multiplying.");
		pOpts->add("-transposeb", "Transpose [b] before multiplying.");
	}
	{
		UsageNode* pNode = pRoot->add("multiplyscalar [dataset] [scalar]", "Multiply all elements in [dataset] by the specified scalar. Results are printed to stdout.");
		pNode->add("[dataset]=in.arff", "The filename of a dataset.");
		pNode->add("[scalar]=0.5", "A scalar to multiply each element by.");
	}
	{
		UsageNode* pNorm = pRoot->add("normalize [dataset] <options>", "Normalize all continuous attributes to fall within the specified range. (Nominal columns are left unchanged.)");
		pNorm->add("[dataset]=data.arff", "The filename of a dataset");
		UsageNode* pOpts = pNorm->add("<options>");
		pOpts->add("-range [min] [max]", "Specify the output min and max values. (The default is 0 1.)");
	}
	{
		UsageNode* pNomToCat = pRoot->add("nominaltocat [dataset] <options>", "Convert all nominal attributes in the data to vectors of real values by representing them as a categorical distribution. Columns with only two nominal values are converted to 0 or 1. If there are three or more possible values, a column is created for each value. The column corresponding to the value is set to 1, and the others are set to 0. (This is similar to Weka's NominalToBinaryFilter.)");
		pNomToCat->add("[dataset]=data.arff", "The filename of a dataset");
		UsageNode* pOpts = pNomToCat->add("<options>");
		pOpts->add("-maxvalues [cap]=8", "Specify the maximum number of nominal values for which to create new columns. If not specified, the default is 12.");
	}
	{
		UsageNode* pPowerCols = pRoot->add("powercolumns [dataset] [column-list] [exponent]", "Raises the values in the specified columns to some power (or exponent).");
		pPowerCols->add("[dataset]=data.arff", "The filename of a dataset.");
		pPowerCols->add("[column-list]=0", "A comma-separated list of zero-indexed columns to transform. A hypen may be used to specify a range of columns. Example: 0,2-5,7");
		pPowerCols->add("[exponent]=0.5", "An exponent value, such as 0.5, 2, etc.");
	}
	pRoot->add("pseudoinverse [dataset]=m.arff", "Compute the Moore-Penrose pseudo-inverse of the specified matrix of real values.");
	pRoot->add("reducedrowechelonform [dataset]=m.arff", "Convert a matrix to reduced row echelon form. Results are printed to stdout.");
	{
		UsageNode* pRotate = pRoot->add("rotate [dataset] [col_x] [col_y] [angle_degrees]","Rotate angle degrees around the origin in in the col_x,col_y plane.  Only affects the values in col_x and col_y.");
		pRotate->add("[dataset]=in.arff", "The filename of a dataset.");
		pRotate->add("[col_x]=0", "The zero-based index of an attribute to serve as the x coordinate in the plane of rotation.  Rotation from x to y will be 90 degrees. col_x must be a real-valued attribute.");
		pRotate->add("[col_y]=1", "The zero-based index of an attribute to serve as the y coordinate in the plane of rotation.  Rotation from y to x will be 270 degrees. col_y must be a real-valued attribute.");
		pRotate->add("[angle_degrees]=90.0","The angle in degrees to rotate around the origin in the col_x,col_y plane.");
	}
	{
		UsageNode* pSampRows = pRoot->add("samplerows [dataset] [portion]", "Samples from the rows in the specified dataset and prints them to stdout. This tool reads each row one-at-a-time, so it is well-suited for reducing the size of datasets that are too big to fit into memory. (Note that unlike most other tools, this one does not convert CSV to ARFF format internally. If the input is CSV, the output will be CSV too.)");
		pSampRows->add("[dataset]=in.arff", "The filename of a dataset. ARFF, CSV, and a few other formats are supported.");
		pSampRows->add("[portion]=0.1", "A value between 0 and 1 that specifies the likelihood that each row will be printed to stdout.");
		UsageNode* pOpts = pSampRows->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pScaleCols = pRoot->add("scalecolumns [dataset] [column-list] [scalar]", "Multiply the values in the specified columns by a scalar.");
		pScaleCols->add("[dataset]=data.arff", "The filename of a dataset.");
		pScaleCols->add("[column-list]=0", "A comma-separated list of zero-indexed columns to transform. A hypen may be used to specify a range of columns. Example: 0,2-5,7");
		pScaleCols->add("[scalar]=0.5", "A scalar value.");
	}
	{
		UsageNode* pShiftCols = pRoot->add("shiftcolumns [dataset] [column-list] [offset]", "Add [offset] to all of the values in the specified columns.");
		pShiftCols->add("[dataset]=data.arff", "The filename of a dataset.");
		pShiftCols->add("[column-list]=0", "A comma-separated list of zero-indexed columns to transform. A hypen may be used to specify a range of columns. Example: 0,2-5,7");
		pShiftCols->add("[offset]=1.0", "A positive or negative value to add to the values in the specified columns.");
	}
	{
		UsageNode* pShuffle = pRoot->add("shuffle [dataset] <options>", "Shuffle the row order.");
		pShuffle->add("[dataset]=data.arff", "The filename of a dataset");
		UsageNode* pOpts = pShuffle->add("<options>");
		pOpts->add("-seed [value]=0", "Specify a seed for the random number generator.");
	}
	{
		UsageNode* pSignif = pRoot->add("significance [dataset] [attr1] [attr2] <options>", "Compute statistical significance values for the two specified attributes.");
		pSignif->add("[dataset]=results.arff", "The filename of a .arff file.");
		pSignif->add("[attr1]=0", "A zero-indexed column number.");
		pSignif->add("[attr2]=0", "A zero-indexed column number.");
		UsageNode* pOpts = pSignif->add("<options>");
		pOpts->add("-tol [value]=0.001", "Sets the tolerance value for the Wilcoxon Signed Ranks test. The default value is 0.001.");
	}
	{
		UsageNode* pSortCol = pRoot->add("sortcolumn [dataset] [col] <options>", "Sort the rows in [dataset] such that the values in the specified column are in ascending order and print the results to to stdout. (The input file is not modified.)");
		pSortCol->add("[dataset]=data.arff", "The filename of a dataset.");
		pSortCol->add("[col]=0", "The zero-indexed column number in which to sort");
		UsageNode* pOpts = pSortCol->add("<options>");
		pOpts->add("-descending", "Sort in descending order instead of ascending order.");
	}
	{
		UsageNode* pSplit = pRoot->add("split [dataset] [rows] [filename1] [filename2] <options>", "Split a dataset into two datasets. (Nothing is printed to stdout.)");
		pSplit->add("[dataset]=data.arff", "The filename of a datset.");
		pSplit->add("[rows]=200", "The number of rows to go into the first file. The rest go in the second file.");
		{
			UsageNode* pOpts = pSplit->add("<options>");
			pOpts->add("-seed [value]", "Specify a seed for the random number generator.");
			pOpts->add("-shuffle","Shuffle the input data before splitting it.");
		}
		pSplit->add("[filename1]=out1.arff", "The filename for one half of the data.");
		pSplit->add("[filename2]=out2.arff", "The filename for the other half of the data.");
	}
	{
		UsageNode* pSplitClass = pRoot->add("splitclass [data] [attr] <options>", "Splits a dataset by a class attribute, such that a separate file is created for each unique class label. The generated filenames will be \"[data]_[value]\", where [value] is the unique class label value.");
		pSplitClass->add("[data]=data.arff", "The filename of a dataset.");
		pSplitClass->add("[attr]=0", "The zero-indexed column number of the class attribute.");
		UsageNode* pOpts = pSplitClass->add("<options>");
		pOpts->add("-dropclass", "Drop the class attribute after splitting the data. (The default is to include the class attribute in each of the output datasets, which is rather redundant since every row in the file will have the same class label.)");
	}
	{
		UsageNode* pSplitFold = pRoot->add("splitfold [dataset] [i] [n] <options>", "Divides a dataset into [n] parts of approximately equal size, then puts part [i] into one file, and the other [n]-1 parts in another file. (This tool may be useful, for example, to implement n-fold cross validation.)");
		pSplitFold->add("[dataset]=data.arff", "The filename of a datset.");
		pSplitFold->add("[i]=0", "The (zero-based) index of the fold, or the part to put into the training set. [i] must be less than [n].");
		pSplitFold->add("[n]=10", "The number of folds.");
		UsageNode* pOpts = pSplitFold->add("<options>");
		pOpts->add("-out [train_filename] [test_filename]", "Specify the filenames for the training and test portions of the data. The default values are train.arff and test.arff.");
	}
	{
		UsageNode* pSD = pRoot->add("squareddistance [a] [b]", "Computesthe sum and mean squared distance between dataset [a] and [b]. ([a] and [b] are each the names of files in .arff format. They must have the same dimensions.)");
		pSD->add("[a]=a.arff", "The filename of a dataset.");
		pSD->add("[b]=b.arff", "The filename of a dataset.");
	}
	{
		UsageNode* pSVD = pRoot->add("svd [matrix] <options>", "Compute the singular value decomposition of a matrix.");
		pSVD->add("[matrix]=m.arff", "The filename of the matrix.");
		UsageNode* pOpts = pSVD->add("<options>");
		pOpts->add("-ufilename [filename]=u.arff", "Set the filename to which U will be saved. U is the matrix in which the columns are the eigenvectors of [matrix] times its transpose. The default is u.arff.");
		pOpts->add("-sigmafilename [filename]=sigma.arff", "Set the filename to which Sigma will be saved. Sigma is the matrix that contains the singular values on its diagonal. All values in Sigma except the diagonal will be zero. If this option is not specified, the default is to only print the diagonal values (not the whole matrix) to stdout. If this options is specified, nothing is printed to stdout.");
		pOpts->add("-vfilename [filename]=v.arff", "Set the filename to which V will be saved. V is the matrix in which the row are the eigenvectors of the transpose of [matrix] times [matrix]. The default is v.arff.");
		pOpts->add("-maxiters [n]=100", "Specify the number of times to iterate before giving up. The default is 100, which should be sufficient for most problems.");
	}
	{
		UsageNode* pSC = pRoot->add("swapcolumns [dataset] [col1] [col2]", "Swap two columns in the specified dataset and prints the results to stdout. (Columns are zero-indexed.)");
		pSC->add("[dataset]=data.arff", "The filename of a dataset");
		pSC->add("[col1]=0", "A zero-indexed column number.");
		pSC->add("[col2]=1", "A zero-indexed column number.");
	}
	{
		UsageNode* pTransition = pRoot->add("transition [action-sequence] [state-sequence] <options>", "Given a sequence of actions and a sequence of states (each in separate datasets), this generates a single dataset to map from action-state pairs to the next state. This would be useful for generating the data to train a transition function.");
		UsageNode* pOpts = pTransition->add("<options>");
		pOpts->add("-delta", "Predict the delta of the state transition instead of the new state.");
	}
	{
		UsageNode* pThresh = pRoot->add("threshold [dataset] [column] [threshold]",
		   "Outputs a copy of dataset such that any value v in the "
		   "given column becomes 0 if v <= threshold and 1 otherwise."
		   "  Only works on continuous attributes.");
		pThresh->add("[dataset]=in.arff", "The filename of a dataset.");
		pThresh->add("[column]=0", "The zero-indexed column number to threshold.");
		pThresh->add("[threshold]=0.5", "The threshold value.");
	}
	{
		pRoot->add("transpose [dataset]=m.arff", "Transpose the data such that columns become rows and rows become columns.");
	}
	{
		pRoot->add("zeromean [dataset]=m.arff","Subtracts the mean from all values "
		   "of all continuous attributes, so that their means in the "
		   "result are zero. Leaves nominal attributes untouched.");
	}
	{
		pRoot->add("usage", "Print usage information.");
	}
	return pRoot;
}


