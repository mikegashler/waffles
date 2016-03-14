/*
  The contents of this file are dedicated by all of its authors, including

    Eric Moyer,
    Michael S. Gashler,

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

#include "GSelfOrganizingMap.h"
#include "GVec.h"
#include "GRand.h"
#include "GMath.h"
#include "GImage.h"
#include "GDistance.h"
#include "GDom.h"
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <set>
#include <map>
#include <algorithm>
#include <memory>

namespace GClasses {

  namespace SOM{

    //virtual
    void GridTopology::setLocations(std::vector<double> outputAxesMax,
				    std::vector<Node>& nodes){
      //Calculate the integer versions of the axes
      std::vector<unsigned> intAxesMax(outputAxesMax.size());
      for(std::size_t dim = 0; dim < outputAxesMax.size(); ++dim){
	int higher = (int)(outputAxesMax[dim]+1.0);
	int lower = (int)outputAxesMax[dim];
	double actualVal = outputAxesMax[dim];
	if(higher - actualVal < epsilon()){
	  intAxesMax[dim] = higher;
	}else{
	  intAxesMax[dim] = lower;
	}
      }

      //Calculate the number of nodes needed and reallocate
      double natsRequired = 0; //Information content in nats rather than bits
      double natsAvailable =
	std::log((double)std::numeric_limits<std::size_t>::max());
      for(std::size_t dim = 0; dim < intAxesMax.size();++dim){
	natsRequired += std::log((double)intAxesMax[dim]);
      }
      if(natsRequired > natsAvailable){
	std::stringstream needed; needed << std::exp(natsRequired);
	throw Ex("Not enough bits to index an array of ", needed.str(),
		   " nodes in SOM::GridTopology.");
      }else{
	std::size_t nodesNeeded = 1;
	for(std::size_t dim = 0; dim < intAxesMax.size();++dim){
	  nodesNeeded *= intAxesMax[dim];
	}
	if(nodes.size() < nodesNeeded){
	  nodes.resize(nodesNeeded);
	}
      }

      //Initialize the node locations
      for(std::size_t n = 0; n < nodes.size(); ++n){
	Node& node = nodes[n];
	std::vector<double> loc(intAxesMax.size());
	std::size_t n_quotient = n;
	for(std::size_t dim = 0; dim < loc.size(); ++dim){
	  loc[dim] = (double)(n_quotient % intAxesMax[dim]);
	  n_quotient = n_quotient / intAxesMax[dim];
	}
	node.outputLocation = loc;
      }
    }


    NodeWeightInitializationTrainingSetSample::
    NodeWeightInitializationTrainingSetSample(GRand& rand)
	:pRand(&rand){}

    //virtual
    void NodeWeightInitializationTrainingSetSample::
    setWeights(std::vector<Node>& nodes,
	       GDistanceMetric& weightDistance,
	       const GMatrix* pIn) const{
      assert(pRand != NULL);
      assert(pIn != NULL);
      assert(pIn->rows() >= nodes.size());

      //Set the weight distance function's relation metadata
      weightDistance.init(pIn->relation().clone(), true);

      //Generate a sample of indices
      std::set<std::size_t> used;
      std::list<std::size_t> inOrder;
      while(used.size() < nodes.size()){
	std::size_t index = (size_t)pRand->next(pIn->rows());
	if(used.count(index) == 0){
	  used.insert(index);
	  inOrder.push_back(index);
	}
      }

      //Copy that sample of indices
      std::list<std::size_t>::const_iterator sourceIdx;
      std::size_t destIdx;
      for(destIdx = 0, sourceIdx = inOrder.begin();
	  sourceIdx != inOrder.end(); ++sourceIdx, ++destIdx){
	const double *begin = pIn->row(*sourceIdx).data();
	const double *end = begin + pIn->cols();
	nodes[destIdx].weights = std::vector<double>(begin, end);
      }
    }


    //virtual
    void ReporterChain::add(Reporter*toAdd){ m_reporters.push_back(toAdd); }

    //virtual
    void ReporterChain::start(const GMatrix* trainingData,
			      int maxIterations, int maxSubIterations){
      for(iterator r = m_reporters.begin(); r != m_reporters.end(); ++r){
	(*r)->start(trainingData, maxIterations, maxSubIterations);
      }
    }

    //virtual
    void ReporterChain::newStatus(unsigned iteration, unsigned subIteration,
				  const GSelfOrganizingMap& map){
      for(iterator r = m_reporters.begin(); r != m_reporters.end(); ++r){
	(*r)->newStatus(iteration, subIteration, map);
      }
    }

    //virtual
    void ReporterChain::stop(unsigned iteration, unsigned subIteration,
			     const GSelfOrganizingMap& map){
      for(iterator r = m_reporters.begin(); r != m_reporters.end(); ++r){
	(*r)->stop(iteration, subIteration, map);
      }
    }

    //virtual
    ReporterChain::~ReporterChain(){
      for(iterator r = m_reporters.begin(); r != m_reporters.end(); ++r){
	delete (*r);
      }
    }

    //virtual
    void SVG2DWeightReporter::start(const GMatrix* trainingData,
				    int maxIterations, int maxSubIterations){
      double log10 = std::log(10.0);
      double iterDigits, subIterDigits;
      if(maxIterations < 0){
	iterDigits = 6;
      }else{
	iterDigits = std::log((double)maxIterations)/log10;
      }
      if(maxSubIterations < 0){
	subIterDigits = 4;
      }else{
	subIterDigits = std::log((double)maxSubIterations)/log10;
      }
      m_totalDigits = std::min(std::numeric_limits<unsigned>::digits10,
				    (int)(iterDigits+subIterDigits+1));
      m_nextNumberToOutput = 1;
      if(m_showTrainingData){
	Point p;
	for(unsigned i = 0; i < trainingData->rows(); ++i){
	  const double* d = trainingData->row(i).data();
	  p.x = d[m_xDim]; p.y = d[m_yDim];
	  m_trainingData.push_back(p);
	}
      }
    }

    namespace{
      ///Helper class for the svg output method representing an
      ///undirected 2D line
      struct SVG2DLine{
	double x1;
	double y1;
	double x2;
	double y2;
	SVG2DLine(double x_1, double y_1, double x_2, double y_2)
	  :x1(x_1),y1(y_1),x2(x_2),y2(y_2){
	  //Sort the points involved in the line lexically so that the
	  //line becomes undirected
	  if(x1 < x2 || (x1 == x2 && y1 < y2)){
	    std::swap(x1,x2); std::swap(y1,y2);
	  }
	}
	///Compare two lines in lexical ordering - used so can put
	///them in a std::set
	bool operator<(const SVG2DLine& rhs) const{
	  if(x1 < rhs.x1){
	    return true;
	  }else if(x1 == rhs.x1){
	    if(y1 < rhs.y1){
	      return true;
	    }else if(y1 == rhs.y1){
	      if(x2 < rhs.x2){
		return true;
	      }else if(x2 == rhs.x2){
		if(y2 < rhs.y2){
		  return true;
		}
	      }
	    }
	  }
	  return false;
	}
      };

      ///Print the svg code for the given line
      std::ostream& operator<<(std::ostream& out, const SVG2DLine& l){
	return
	  out << "<line x1=\"" << l.x1 << "\" y1=\"" << l.y1
	      <<    "\" x2=\"" << l.x2 << "\" y2=\"" << l.y2
	      << "\"/>";
      }

      class BoundingBox{
	inline double inf() const{
	  return std::numeric_limits<double>::infinity();
	}
      public:
	///Coordinates of the axis aligned 2D bounding box
	double minX; double minY; double maxX; double maxY;

	///Create an uninitialized bounding box
	BoundingBox():
	  minX(inf()), minY(inf()), maxX(-inf()), maxY(-inf()){}

	double height(){ return (maxY >= minY)?maxY-minY:0; }
	double width(){ return (maxX >= minX)?maxX-minX:0; }

	///Add the lines in the stl container to the bounding box
	template <class InputIter>
	void addLines(InputIter begin, InputIter end){
	  while(begin != end){
	    if(minX > begin->x1){ minX = begin->x1; }
	    if(minY > begin->y1){ minY = begin->y1; }
	    if(minX > begin->x2){ minX = begin->x2; }
	    if(minY > begin->y2){ minY = begin->y2; }

	    if(maxX < begin->x1){ maxX = begin->x1; }
	    if(maxY < begin->y1){ maxY = begin->y1; }
	    if(maxX < begin->x2){ maxX = begin->x2; }
	    if(maxY < begin->y2){ maxY = begin->y2; }
	    ++begin;
	  }
	}

	///Add the points in the stl container to the bounding box
	template <class InputIter>
	void addPoints(InputIter begin, InputIter end){
	  while(begin != end){
	    if(minX > begin->x){ minX = begin->x; }
	    if(minY > begin->y){ minY = begin->y; }
	    if(maxX < begin->x){ maxX = begin->x; }
	    if(maxY < begin->y){ maxY = begin->y; }
	    ++begin;
	  }
	}

      };

    }

    unsigned SVG2DWeightReporter::neighborsNeeded
    (const GSelfOrganizingMap& map){
      //TODO: write neighborsNeeded correctly so it will work with random topology.  It should store a graph as a multimap<size_t,size_t> node number to node number.  Build that graph using successively larger sets of neighbors until it is connected.
      return 1;
    }


    //virtual
    void SVG2DWeightReporter::newStatus(unsigned iteration,
					unsigned subIteration,
					const GSelfOrganizingMap& map){
      //Create the output file (and update the next number to use)
      std::stringstream fnBuf;
      fnBuf << m_baseFilename << "_"
	    << std::setfill('0') << std::setw(m_totalDigits)
	    << (m_nextNumberToOutput++) << ".svg";
      const std::string filename = fnBuf.str();
      std::ofstream out(filename.c_str());
      if(!out){
	std::cerr << "Warning: Self-organizing map reporter could not write "
		  << "to the file named \"" << filename
		  << "\".  Continuing anyway." << std::endl;
	return;
      }

      //For each node, add a line between it and its nearest neighbors
      //to the set of lines to draw.
      std::vector<Node> nodes = map.nodes();
      if(nodes.size() > 0){
	std::size_t numWeights = nodes.at(1).weights.size();
	if(numWeights <= m_xDim){
	  throw Ex("X dimension is larger than the number of weights in "
		     "SVG output.");
	}
	if(numWeights <= m_yDim){
	  throw Ex("Y dimension is larger than the number of weights in "
		     "SVG output.");
	}
      }
      unsigned neighborsToReq = neighborsNeeded(map);
      std::set<SVG2DLine> lines;
      for(std::size_t i = 0; i < nodes.size(); ++i){
	double x1 = nodes.at(i).weights.at(m_xDim);
	double y1 = nodes.at(i).weights.at(m_yDim);
	std::vector<std::size_t> neighbors =
	  map.nearestNeighbors((unsigned int)i, neighborsToReq);
	for(std::vector<std::size_t>::const_iterator n = neighbors.begin();
	    n != neighbors.end(); ++n){
	  double x2 = nodes.at(*n).weights.at(m_xDim);
	  double y2 = nodes.at(*n).weights.at(m_yDim);
	  lines.insert(SVG2DLine(x1,y1,x2,y2));
	}
      }

      //Calculate the bounding box of the lines
      BoundingBox bb;
      bb.addLines(lines.begin(), lines.end());
      if(m_showTrainingData){
	bb.addPoints(m_trainingData.begin(), m_trainingData.end());
      }

      //TODO: change SVG in a way that makes the resulting diagrams a reasonable size - like 3 inches - no matter what input was used.

      //Write the svg header
      out << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
	  << "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
	  << "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
	  << "<svg viewbox=\""<< bb.minX << " " << bb.minY << " "
	  << bb.width() << " " << bb.height()
	  << "\" preserveAspectRatio=\"xMinYMin\"\n"
	  << "     width=\""<< bb.width() << "px\" height=\""
	  << bb.height() << "px\"\n"
	  << "     xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
	  << "<desc>Self organizing map training visualization using a \n"
	  << "      2D mesh.  Iteration:" << iteration
	  << " Sub-iteration:" << subIteration << "\n"
	  << "      Original filename: \"" << filename << "\"\n"
	  << "      Plotting input dimensions " << m_xDim << " and "
	  << m_yDim << " as (x,y) coordinates.\n"
	  << "</desc>\n";

      //Translate the data so it is all positive
      out << "<g transform=\"translate(" << -bb.minX <<","<< -bb.minY
	  << ")\">\n";

      //If show training data, write the points.  Give them radius of bb/100
      if(m_showTrainingData){
	out << "<g stroke=\"none\" fill=\"red\">\n";
	std::stringstream rBuf; rBuf << std::scientific << std::setprecision(2)
				<< (bb.height()/100);
	std::string r = rBuf.str();
	for(std::vector<Point>::const_iterator it = m_trainingData.begin();
	    it != m_trainingData.end(); ++it){
	  out << "<circle cx=\"" << it->x << "\""
	      << " cy=\"" << it->y << "\""
	      << " r=\"" << r << "\""
	      << "/>\n";
	}
	out << "</g>\n";
      }

      //Write the lines - give them a width of bb/1000
      std::stringstream strokeW; strokeW << (bb.height()/1000);
      out  << "<g stroke=\"black\" stroke-width=\""<< strokeW.str() <<"\">\n";
      for(std::set<SVG2DLine>::const_iterator it = lines.begin();
	  it != lines.end(); ++it){
	out << (*it) << "\n";
      }
      out << "</g>\n";

      //End the translation
      out << "</g>\n";

      //End the svg file
      out << "</svg>\n";
    }

    //virtual
    void SVG2DWeightReporter::stop(unsigned iteration, unsigned subIteration,
				   const GSelfOrganizingMap& map){
      newStatus(iteration, subIteration, map);
    }


    //virtual
    double GaussianWindowFunction::
    operator()(double width, double distance) const{
      const double stddev = distance/width;
      return (stddev < truncationSigmas)?
	std::exp(-0.5*(stddev)*(stddev))
	:0;
    }


    BatchTraining::BatchTraining
    (double initialNeighborhoodSize,
     double finalNeighborhoodSize,
     unsigned numIterations,
     unsigned maxSubIterationsBeforeChangingNeighborhood,
     NodeWeightInitialization* weightInitialization,
     NeighborhoodWindowFunction* windowFunc,
     Reporter* reporter):
      m_initialNeighborhoodSize(initialNeighborhoodSize),
      m_timeFactor(std::log(finalNeighborhoodSize/initialNeighborhoodSize)
		   /(numIterations-1)),
      m_numIterations(numIterations),
      m_maxSubIterationsBeforeChangingNeighborhood
      (maxSubIterationsBeforeChangingNeighborhood),
      m_weightInitialization(weightInitialization),
      m_windowFunc(windowFunc),
      m_reporter(reporter)
    {
      assert(initialNeighborhoodSize >= 0);
      assert(finalNeighborhoodSize >= 0);
      assert(numIterations > 0);
      assert(maxSubIterationsBeforeChangingNeighborhood > 0);
      assert(weightInitialization != NULL);
      assert(windowFunc != NULL);
      assert(reporter != NULL);
    }

    namespace{
      class IncrementalWeightedAverage{
	std::vector<double> weightedSum;
	double totalWeights;
	typedef std::vector<double>::const_iterator cIter;
	typedef std::vector<double>::iterator        iter;
	bool sameSinceLastReset;
      public:
	///Create an IncrementalWeightedAverage object that takes an
	///average of vectors of numDimensions dimensions
	IncrementalWeightedAverage(unsigned numDimensions)
	  :weightedSum(numDimensions,0), totalWeights(0)
	  ,sameSinceLastReset(true)
	{}

	///Return the average so far.  Should not be called if the
	///incremental average has not been updated with an add since
	///the last reset (object creation counts as a reset).  Also
	///should not be called if the sum of the weights is 0.
	std::vector<double> average() const{
	  if(sameSinceLastReset){
	    throw Ex("Average called on IncrementalWeightedAverage ",
		       "object that has not been updated since the last ",
		       "reset (or since object creation)");
	  }
	  assert(totalWeights != 0);
	  std::vector<double> out = weightedSum;
	  for(iter i = out.begin(); i != out.end(); ++i){ (*i)/=totalWeights; }
	  return out;
	}

	///Set the sums and weights to 0
	void reset(){
	  sameSinceLastReset = true;
	  totalWeights = 0;
	  std::fill(weightedSum.begin(), weightedSum.end(), 0);
	}

	///Return true if add has been called since the last reset (or
	///since object creation if there have not been any resets)
	bool changedSinceLastReset(){ return !sameSinceLastReset; }

	///Add another value with the given weight.  It is assumed
	///that the value array has at least as many entries as the
	///weightedSum vector
	void add(double weight, const double *value){
	  sameSinceLastReset = false;
	  totalWeights += weight;
	  const double* val=value;
	  iter sum=weightedSum.begin();
	  for(; sum != weightedSum.end(); ++val,++sum){
	    (*sum)+=weight * (*val);
	  }
	}
      };

      ///Resets a weighed average object
      struct Reset{
	void operator()(IncrementalWeightedAverage& a){ a.reset(); }
      };
    }//anonymous namespace

    //virtual
    void BatchTraining::train(GSelfOrganizingMap& map, const GMatrix* pIn){
      assert(pIn != NULL);
      //Report start
      m_reporter->start(pIn, m_numIterations,
			m_maxSubIterationsBeforeChangingNeighborhood);

      //Initialize weights
      m_weightInitialization->setWeights(map.nodes(), weightDistance(map), pIn);

      //Set the before relation
      setPRelationBefore(map, pIn->relation());

#if 0
      //DEBUG - print the node locations and 1st 5 neighbors within
      //startwidth distance
      {
	std::ofstream out ("nodeloc.txt");
	for(unsigned i = 0; i < map.nodes().size(); ++i){
	  out << "Node " << i << ": ";
	  SOM::Node n = map.nodes()[i];
	  for(unsigned j = 0; j < n.outputLocation.size(); ++j){
	    out << n.outputLocation[j] << " ";
	  }
	  out << "\n";
	  out << "Node " << i << " neighbors: ";
	  std::vector<NodeAndDistance> nei = map.neighborsInCircle
	    (i, m_windowFunc->minZeroDistance(m_initialNeighborhoodSize));
	  for(unsigned j = 0; j < 5; ++j){
	    out << nei[j].nodeIdx << "@" << nei[j].distance << " ";
	  }
	  out << "\n";
	}
      }
#endif
      //Create weighted averages for each node
      std::vector<IncrementalWeightedAverage>
	aves(map.nodes().size(), IncrementalWeightedAverage((unsigned int)pIn->cols()));

      //Create list of which node was closest to a given data point
      //last time - initializing to a non-existent node index
      std::size_t invalidIdx = std::numeric_limits<std::size_t>::max();
      assert(invalidIdx >= pIn->rows()); //Ensure it is really invalid
      std::vector<std::size_t> prevClosest(pIn->rows(), invalidIdx);

      //Report before first iteration
      m_reporter->newStatus(0, 0, map);

      //For each major iteration, train as close to convergence as possible
      for(unsigned superepoch = 1; superepoch <= m_numIterations; ++superepoch){
	//calculate neighborhood width
	double width = m_initialNeighborhoodSize*
	  std::exp(m_timeFactor*(superepoch-1));
	//update weights until convergence or run out of time
	for(unsigned epoch = 0;
	    epoch < m_maxSubIterationsBeforeChangingNeighborhood; ++epoch){
	  //assume that converged
	  bool hasConverged = true;
	  //For each input point
	  for(std::size_t rIdx = 0; rIdx < pIn->rows(); ++rIdx){
	    const GVec& pRow = pIn->row(rIdx);
	    //Find the closest node
	    std::size_t best = map.bestMatch(pRow);
	    //compare to closest last time - if different, not converged
	    ///OPTIMIZE: could try not zeroing the averages and then
	    ///updating only places where closest point had changed -
	    ///but doing it twice, first subtracting the old and then
	    ///adding the new - introduces cumulative rounding error
	    if(prevClosest[rIdx] != best){
	      hasConverged = false;
	      prevClosest[rIdx] = best;
	    }
	    //For each neighbor within range of best so that the
	    //window function is non-zero, add weighted input point to
	    //incremental weighted average
	    std::vector<SOM::NodeAndDistance> nbrs =
	      map.neighborsInCircle((unsigned int)best, m_windowFunc->minZeroDistance(width));
	    std::vector<SOM::NodeAndDistance>::const_iterator nItr;
	    for(nItr = nbrs.begin(); nItr != nbrs.end(); ++nItr){
	      double weight = (*m_windowFunc)(width, nItr->distance);
	      aves[nItr->nodeIdx].add(weight, pRow.data());
	    }
	    //Add the point to the best point's average also
	    aves[best].add((*m_windowFunc)(width, 0), pRow.data());
	  }
	  //Set all nodes' weights to final weighted average and zero
	  //averages for next pass.  Any nodes that were not in the
	  //neighborhood of any winners are left unchanged.
	  std::vector<SOM::Node>::iterator nIter = map.nodes().begin();
	  std::vector<IncrementalWeightedAverage>::iterator aIter
	    = aves.begin();
	  for(; aIter != aves.end(); ++nIter, ++aIter){
	    if(aIter->changedSinceLastReset()){
	      nIter->weights = aIter->average();
	      aIter->reset();
	    }
	  }
	  //Report end of iteration.
	  m_reporter->newStatus(superepoch, epoch, map);
	  //if converged, start another major iteration
	  if(hasConverged){ break; }
	}
      }
      //Report stop
      m_reporter->newStatus(m_numIterations+1, 0, map);
    }

    //virtual
    BatchTraining::~BatchTraining(){
      delete m_weightInitialization;
      delete m_windowFunc;
      delete m_reporter;
    }


    TraditionalTraining::TraditionalTraining
    (double initialWidth, double finalWidth,
     double initialRate, double finalRate,
     unsigned numIterations,
     NodeWeightInitialization* weightInitialization,
     NeighborhoodWindowFunction* windowFunc,
     Reporter* reporter):
      m_rand(0),
      m_initialWidth(initialWidth), m_finalWidth(finalWidth),
      m_widthFactor(std::log(finalWidth/initialWidth)/(numIterations-1)),
      m_initialRate(initialRate), m_finalRate(finalRate),
      m_rateFactor(std::log(finalRate/initialRate)/(numIterations-1)),
      m_numIterations(numIterations),
      m_weightInitialization(weightInitialization),
      m_windowFunc(windowFunc),
      m_reporter(reporter){
      assert(m_initialWidth >= 0);
      assert(m_finalWidth >= 0);
      assert(m_initialRate >= 0);
      assert(m_finalRate >= 0);
      assert(m_numIterations > 0);
      assert(m_weightInitialization != NULL);
      assert(m_windowFunc != NULL);
      assert(m_reporter != NULL);
    }


    namespace{
      ///Increment the given weights by a scale factor times a vector
      ///from the weights to their destination
      void moveWeightsCloser(std::vector<double>&weights,
			    const double* destination, double scale){
	std::vector<double>::iterator w = weights.begin();
	while(w != weights.end()){
	  double inc = *destination - *w;
	  (*w) += scale * inc;
	  ++destination; ++w;
	}
      }

    }

    //TODO: make all training algorithms take a pointer to a const
    //GMatrix (this will require making GMatrix const-correct)

    //virtual
    void TraditionalTraining::train(GSelfOrganizingMap& map, const GMatrix* pInput){
      assert(pInput != NULL);
      //Report start
      m_reporter->start(pInput, m_numIterations, 1);

      //Initialize weights
      m_weightInitialization->setWeights(map.nodes(), weightDistance(map),
					 pInput);

      //Set the before relation
      setPRelationBefore(map, pInput->relation());

      //Report before first iteration
      m_reporter->newStatus(0, 0, map);

      //Copy the input
      GMatrix* pInputCopy = new GMatrix();
      pInputCopy->copy(pInput);
      std::unique_ptr<GMatrix> inputCopy(pInputCopy);

      //For each iteration
      for(unsigned iteration = 0; iteration < m_numIterations; ++iteration){
	//Calculate the width and learning rate
	double width = 	m_initialWidth*std::exp(m_widthFactor*(iteration));
	double rate = 	m_initialRate*std::exp(m_rateFactor*(iteration));

	//If starting a run through the list of input points, shuffle
	//them
	if(iteration % inputCopy->rows() == 0){
	  inputCopy->shuffle(m_rand); }
	const GVec& point = inputCopy->row(iteration % inputCopy->rows());

	//Find the best match to the current input point
	std::size_t best = map.bestMatch(point);

	//Move the best match and its neighbors closer to the input
	std::vector<SOM::NodeAndDistance> nbrs =
	  map.neighborsInCircle((unsigned int)best, m_windowFunc->minZeroDistance(width));
	std::vector<SOM::NodeAndDistance>::const_iterator nItr;
	for(nItr = nbrs.begin(); nItr != nbrs.end(); ++nItr){
	  moveWeightsCloser(map.nodes()[nItr->nodeIdx].weights, point.data(),
			   rate*(*m_windowFunc)(width, nItr->distance));
	}
	moveWeightsCloser(map.nodes()[best].weights, point.data(),
			 rate*(*m_windowFunc)(width, 0));

	//Report iteration
	m_reporter->newStatus(iteration+1, 1, map);
      }
      //Report stopping
      m_reporter->stop(m_numIterations, 1, map);
    }


    //virtual
    TraditionalTraining::~TraditionalTraining(){
      delete m_weightInitialization;
      delete m_windowFunc;
      delete m_reporter;
    }


  } //Namespace SOM


namespace{
  //Add a double vector to the given document and return the node created
  GDomNode* doubleVectorSerialize(GDom* pDoc, const std::vector<double>& v){
    GDomNode* pNode = pDoc->newList();
    std::vector<double>::const_iterator it;
    for(it = v.begin(); it != v.end(); ++it){
      pNode->addItem(pDoc, pDoc->newDouble(*it));
    }
    return pNode;
  }

  std::vector<double> doubleVectorDeserialize(const GDomNode* pNode){
    GDomListIterator it(pNode);
    std::size_t size = it.remaining();
    std::vector<double> out; out.reserve(size);
    for(; it.current(); it.advance()){
      out.push_back(it.current()->asDouble());
    }
    assert(out.size() == size);
    return out;
  }

  ///Serialize a vector containing some object that has a serialize method
  template<class vectortype>
  GDomNode* objVectorSerialize(GDom* pDoc, const vectortype& v){
    GDomNode* pNode = pDoc->newList();
    typename vectortype::const_iterator it;
    for(it = v.begin(); it != v.end(); ++it){
      pNode->addItem(pDoc, it->serialize(pDoc));
    }
    return pNode;
  }

  ///Deserialize a vector of any object that can be constructed from a
  ///GDomNode
  template<class vectortype>
  vectortype objVectorDeserialize(const GDomNode* pNode){
    GDomListIterator it(pNode);
    std::size_t size = it.remaining();
    vectortype out; out.reserve(size);
    for(; it.current(); it.advance()){
      out.push_back(typename vectortype::value_type(it.current()));
    }
    assert(out.size() == size);
    return out;
  }
}

SOM::Node::Node(const GDomNode* pNode)
  :outputLocation(doubleVectorDeserialize(pNode->field("outputLocation"))),
   weights(doubleVectorDeserialize(pNode->field("weights"))){}


GDomNode* SOM::Node::serialize(GDom*pDoc) const{
  GDomNode* pNode = pDoc->newObj();
  pNode->addField(pDoc, "outputLocation",
		  doubleVectorSerialize(pDoc, outputLocation));
  pNode->addField(pDoc, "weights", doubleVectorSerialize(pDoc, weights));
  return pNode;
}

//virtual
GDomNode* SOM::TrainingAlgorithm::serialize(GDom* pDoc) const{
  GDomNode* pNode = pDoc->newObj();
  pNode->addField(pDoc, "class",
		  pDoc->newString("DummyTrainingAlgorithm"));
  return pNode;
}

SOM::TrainingAlgorithm* SOM::TrainingAlgorithm::deserialize(const GDomNode* pNode){
  return new DummyTrainingAlgorithm();
}


//virtual
GDomNode* GSelfOrganizingMap::serialize(GDom* pDoc) const{
  GDomNode* pNode = baseDomNode(pDoc, "GSelfOrganizingMap");
  pNode->addField(pDoc, "inputDims", pDoc->newInt(m_nInputDims));
  pNode->addField(pDoc, "outputAxes",doubleVectorSerialize(pDoc, m_outputAxes));
  pNode->addField(pDoc, "trainer",m_pTrainer->serialize(pDoc));
  pNode->addField(pDoc, "weightDistance",m_pWeightDistance->serialize(pDoc));
  pNode->addField(pDoc, "nodeDistance", m_pNodeDistance->serialize(pDoc));
  pNode->addField(pDoc, "nodes", objVectorSerialize(pDoc, m_nodes));

  //Do not serialize sorted neighbors, just mark as invalid on
  //deserialization
  return pNode;
}

GSelfOrganizingMap::GSelfOrganizingMap(const GDomNode* pNode) : GIncrementalTransform(pNode) {
  m_nInputDims=(unsigned int)pNode->field("inputDims")->asInt();
  m_outputAxes=doubleVectorDeserialize(pNode->field("outputAxes"));
  m_pTrainer=SOM::TrainingAlgorithm::deserialize(pNode->field("trainer"));
  m_pWeightDistance=GDistanceMetric::deserialize
    (pNode->field("weightDistance"));
  m_pNodeDistance=GDistanceMetric::deserialize
    (pNode->field("nodeDistance"));

  typedef std::vector<SOM::Node> nodevec;
  m_nodes=objVectorDeserialize<nodevec>(pNode->field("nodes"));
  m_sortedNeighborsIsValid = false;
}


GSelfOrganizingMap::GSelfOrganizingMap
(int nMapDims, int nMapWidth, GRand& rand, SOM::Reporter *r)
  : GIncrementalTransform(),
    m_nInputDims(0), //Initialized in training
    m_outputAxes(nMapDims, nMapWidth-1),
    m_pTrainer(new SOM::BatchTraining
	       (2*nMapWidth, 1, 100, 1, //DEBUG Change subiterations back to 100, iterations back to 100 and min neighborhood back to 1 after debugging
		new SOM::NodeWeightInitializationTrainingSetSample(rand),
		new SOM::GaussianWindowFunction(), r)),
    m_pWeightDistance(new GRowDistance()),
    m_pNodeDistance(new GRowDistance()),
    m_nodes((unsigned int)std::pow((double)nMapWidth,(double)nMapDims)),
    m_sortedNeighborsIsValid(false)
{
  //Set the node locations in a grid topology
  SOM::GridTopology g;
  g.setLocations(m_outputAxes, m_nodes);

  //Set the weight vectors to have one identical 0 value each - the
  //training algorithms will change this, but I'd like transform on
  //untrained networks just to return nonsense, not blow up.
  std::vector<double> initialWeights(1,0);
  for(std::vector<SOM::Node>::iterator it = m_nodes.begin();
      it != m_nodes.end(); ++it){
    it->weights = initialWeights;
  }

  //Initialize the distance metrics to use all continuous attributes
  m_pNodeDistance->init(new GUniformRelation(m_outputAxes.size()), true);
  m_pWeightDistance->init(new GUniformRelation(1), true);
}

GSelfOrganizingMap::GSelfOrganizingMap(const std::vector<double>& output_Axes,
				       std::size_t numNodes,
				       SOM::NodeLocationInitialization
				       *topology,
				       SOM::TrainingAlgorithm* trainer,
				       GDistanceMetric* weightDist,
				       GDistanceMetric* nodeDist)
  :m_nInputDims(0), m_outputAxes(output_Axes), m_pTrainer(trainer),
   m_pWeightDistance(weightDist), m_pNodeDistance(nodeDist),
   m_nodes(numNodes),
   m_sortedNeighborsIsValid(false){

  //Set the topology
  topology->setLocations(output_Axes, m_nodes);
  delete topology;

  //Set the weight vectors to have one identical 0 value each - the
  //training algorithms will change this, but I'd like transform on
  //untrained networks just to return nonsense, not blow up.
  std::vector<double> initialWeights(1,0);
  for(std::vector<SOM::Node>::iterator it = m_nodes.begin();
      it != m_nodes.end(); ++it){
    it->weights = initialWeights;
  }

  //Initialize the distance metrics to use all continuous attributes
  m_pNodeDistance->init(new GUniformRelation(m_outputAxes.size()), true);
  m_pWeightDistance->init(new GUniformRelation(1), true);

}


GSelfOrganizingMap::~GSelfOrganizingMap()
{
  delete m_pTrainer;
  delete m_pWeightDistance;
  delete m_pNodeDistance;
}


GMatrix* GSelfOrganizingMap::reduce(const GMatrix& in)
{
  // Train the map on the input
  train(in);

  // Make the output location
  GMatrix* pOut = new GMatrix(in.rows(), outputDimensions());
  std::unique_ptr<GMatrix> hOut(pOut);

  // Transform the input, putting it in the output
  for(std::size_t i = 0; i < in.rows(); ++i){
    transform(in.row(i), pOut->row(i));
  }
  return hOut.release();
}



void GSelfOrganizingMap::transform(const GVec& in, GVec& out){
  unsigned int idx = (unsigned int)bestMatch(in);
  const SOM::Node& n = nodes().at(idx);
  std::copy(n.outputLocation.begin(), n.outputLocation.end(), out.data());
}



std::size_t GSelfOrganizingMap::bestMatch(const GVec& in) const{
  //Eventually this should use a cached k-d tree for low dimensional
  //spaces
  using SOM::Node;
  typedef std::vector<Node>::const_iterator NIter;
  assert(nodes().size() > 0);
  std::size_t best_Match = nodes().size() + 1;
  double bestDistance = std::numeric_limits<double>::infinity();
  for(NIter cur = nodes().begin(); cur != nodes().end(); ++cur){
    const double *weights = &(cur->weights.front());
    {
      GVecWrapper w(weights, cur->weights.size());
      double dissim = (*m_pWeightDistance)(w.vec(), in);
      if(dissim < bestDistance || best_Match >= nodes().size()){
        best_Match = cur - nodes().begin();
        bestDistance = dissim;
      }
    }
  }
  return best_Match;
}

std::vector<std::size_t> GSelfOrganizingMap::bestData
(const GMatrix*data) const{
  typedef std::vector<SOM::Node>::const_iterator NIter;
  if(data == NULL){
    throw Ex("Null data pointer passed to bestData.");
  }
  if(data->rows() == 0){
    throw Ex("No bestData indices can be returned for an empty dataset.");
  }
  using std::vector;
  using std::size_t;
  vector<size_t> out; out.reserve(nodes().size());
  for(NIter cur = nodes().begin(); cur != nodes().end(); ++cur){
    const double *weights = &(cur->weights.front());
    GVecWrapper w(weights, cur->weights.size());
    double bestDist=(*m_pWeightDistance)(w.vec(), data->row(0));
    size_t bestIdx=0;
    for(unsigned dataIdx = 1; dataIdx < data->rows(); ++dataIdx){
      double dist = (*m_pWeightDistance)(w.vec(), data->row(dataIdx));
      if(dist < bestDist){
	bestIdx = dataIdx;
	bestDist = dist;
      }
    }
    out.push_back(bestIdx);
  }
  return out;
}




void GSelfOrganizingMap::regenerateSortedNeighbors() const{
  using SOM::NodeAndDistance;
  //Erase the current sorted neighbors
  m_sortedNeighbors.erase(m_sortedNeighbors.begin(), m_sortedNeighbors.end());
  m_sortedNeighbors.reserve(m_nodes.size());

  const GDistanceMetric* metric = m_pNodeDistance;
  assert(metric != NULL); //If can't maintain this assertion, allocate
			  //a temporary euclidean metric to use

  //curDists is a temporary vector to be filled in turn with each
  //node's distances before they are sorted and added to the end of
  //m_sortedNeighbors
  std::vector<NodeAndDistance> curDists(m_nodes.size()-1,NodeAndDistance(0,-1));

  //For each node, loop through all other nodes, filling in curDists
  //one entry at a time (through dist).  Then sort curDists and add a
  //copy to the end of m_sortedNeighbors
  for(std::size_t curIdx = 0; curIdx < m_nodes.size(); ++curIdx){
    //Fill curDists
    std::vector<NodeAndDistance>::iterator dist = curDists.begin();
    if(dist != curDists.end()){
      for(std::size_t other = 0; other < m_nodes.size(); ++other){
	if(other == curIdx) continue;
	dist->nodeIdx = other;
	GVecWrapper a(m_nodes[curIdx].outputLocation.data(), m_nodes[curIdx].outputLocation.size());
	GVecWrapper b(m_nodes[other].outputLocation.data(), m_nodes[other].outputLocation.size());
	dist->distance = (*metric)(a.vec(), b.vec());
	++dist;
      }
    }
    assert(dist == curDists.end());
    //Sort curDists
    std::sort(curDists.begin(), curDists.end());

    //Add it to end of m_sortedNeighbors
    m_sortedNeighbors.push_back(curDists);
  }
  assert(m_sortedNeighbors.size() == m_nodes.size());
  m_sortedNeighborsIsValid = true;
}


std::vector<std::size_t> GSelfOrganizingMap::nearestNeighbors
(unsigned nodeIdx,
 unsigned numNeighbors,
 double epsilon) const{
  std::vector<std::size_t> out;
  //Take care of border cases in numNeighbors that would produce an
  //empty list or out-of-bounds array accesses
  if(numNeighbors == 0){  return out; }
  if(numNeighbors >= nodes().size()){ numNeighbors = (unsigned int)nodes().size()-1; }
  if(numNeighbors == 0){  return out; }

  //Reserve space for the returned neighbors -- take care of up to
  //hexagonal neighborhoods with 1 neighbor requested but 6 returned
  //without a reallocation
  out.reserve(numNeighbors+5);

  const std::vector<SOM::NodeAndDistance>& neighbors =
    sortedNeighbors().at(nodeIdx);
  assert(numNeighbors <neighbors.size());
  {
    //Copy the requested number of neighbors
    unsigned i;
    for(i = 0; i < numNeighbors; ++i){
      out.push_back(neighbors[i].nodeIdx);
    }
    //Determine what distances are equivalent to the maximum distance
    //of one of the requested neighbors
    assert(i>=1);
    double maxDist = neighbors[i-1].distance + epsilon;

    //Copy all neighbors with equivalent distances
    for(; i < neighbors.size(); ++i){
      if(neighbors[i].distance <= maxDist){
	out.push_back(neighbors[i].nodeIdx);
      }else{
	break;
      }
    }
  }
  return out;
}

std::vector<SOM::NodeAndDistance>
GSelfOrganizingMap::neighborsInCircle(unsigned nodeIdx, double radius) const{
  //Get the neighbors of this node sorted by their distance from it
  assert(nodeIdx < sortedNeighbors().size());
  const std::vector<SOM::NodeAndDistance>& neighbors =
    sortedNeighbors()[nodeIdx];

  //Find the first entry in neighbors that does not have a qualifying
  //distance
  std::vector<SOM::NodeAndDistance>::const_iterator end = neighbors.begin();
  while(end != neighbors.end() && end->distance < radius){ ++end; }

  //Copy and return all entries before the first non-qualifying one
  std::vector<SOM::NodeAndDistance> out(neighbors.begin(), end);
  return out;
}



#ifndef NO_TEST_CODE
#include "GRand.h"
#include "GImage.h"
  namespace{
/*    ///The original test code - which I may still make use of some time
    void originalTest()
    {
      // Make a dataset of random colors
      GRand prng(0);
      GMatrix dataIn(1000, 3);
      int i;
      for(i = 0; i < 1000; i++)
	{
	  GVec& pVec = dataIn[i];
	  pVec[0] = prng.uniform();
	  pVec[1] = prng.uniform();
	  pVec[2] = prng.uniform();
	}

      // Make the map
      GSelfOrganizingMap som(2, 30, prng);
      som.train(dataIn);

      // Make an image of the map
      GImage image;
      image.setSize(30, 30);
      int x, y;
      for(y = 0; y < 30; y++) {
	for(x = 0; x < 30; x++){
	  double* pVec = &(som.nodes()[30 * y + x].weights.front());
	  image.setPixel(x, y, gARGB(0xff, ClipChan((int)(pVec[0] * 256)), ClipChan((int)(pVec[1] * 256)), ClipChan((int)(pVec[2] * 256))));
	}
      }
      //image.SavePNGFile("som.png");
    }
*/
    #include "GSelfOrganizingMapTestData.cpp"

  }//Anonymous namespace

void GSelfOrganizingMap::test(){
  bool generateReports = false;
  GRand rand(1212);
  SOM::Reporter* pReporter;
  if(generateReports)
    pReporter = new SOM::SVG2DWeightReporter("square_som_test",0,1,true);
  else
    pReporter = new SOM::ReporterChain();
  GSelfOrganizingMap squareSom(2, 10, rand, pReporter);
  GMatrix uniformSquareMat(0,2);
  for(unsigned i = 0; i < 170; ++i){
    uniformSquareMat.newRow().set(uniformSquare[i], 2);
  }
  squareSom.train(uniformSquareMat);

  std::unique_ptr<SOM::ReporterChain> cr(new SOM::ReporterChain());
  if(generateReports) {
    cr->add(new SOM::SVG2DWeightReporter("cylinder_som_test_01",0,1,true));
    cr->add(new SOM::SVG2DWeightReporter("cylinder_som_test_02",0,2,true));
    cr->add(new SOM::SVG2DWeightReporter("cylinder_som_test_12",1,2,true));
  }
  GSelfOrganizingMap cylinderSom(2, 10, rand, cr.release());

  GMatrix uniformCylinderMat(0,3);
  for(unsigned i = 0; i < 170; ++i){
    uniformCylinderMat.newRow().set(uniformCylinder[i], 3);
  }
  cylinderSom.train(uniformCylinderMat);


}
#endif // !NO_TEST_CODE

} // namespace GClasses

