/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Michael R. Smith,
    Stephen Ashmore,
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

#include "GBlockWeightless.h"
#ifndef MIN_PREDICT
#include "GMath.h"
#endif // MIN_PREDICT
#include "GActivation.h"
#ifndef MIN_PREDICT
#include "GDistribution.h"
#endif // MIN_PREDICT
#include "GError.h"
#include "GRand.h"
#include "GVec.h"
#include "GDom.h"
#ifndef MIN_PREDICT
#include "GHillClimber.h"
#endif  // MIN_PREDICT
#include "GTransform.h"
#ifndef MIN_PREDICT
#include "GSparseMatrix.h"
#include "GDistance.h"
#include "GAssignment.h"
#endif // MIN_PREDICT
#include "GHolders.h"
#include "GBits.h"
#include "GFourier.h"
#include <memory>

using std::vector;
using std::ostream;

namespace GClasses {


GBlockProductPooling::GBlockProductPooling(size_t outputs, size_t inputs)
: GBlockWeightless()
{
	resize(inputs, outputs);
}

GBlockProductPooling::GBlockProductPooling(GDomNode* pNode)
: GBlockWeightless(pNode), m_outputCount(pNode->field("outputs")->asInt())
{
}

GBlockProductPooling::~GBlockProductPooling()
{
}

GDomNode* GBlockProductPooling::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "outputs", pDoc->newInt(m_outputCount));
	return pNode;
}

// virtual
std::string GBlockProductPooling::to_str() const
{
	std::ostringstream os;
	os << "[GBlockProductPooling:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "]\n";
	return os.str();
}

void GBlockProductPooling::resize(size_t inputCount, size_t outputCount)
{
	if(outputCount * 2 != inputCount)
		throw Ex("inputCount must be 2*outputCount");
	m_outputCount = outputCount;
}

// virtual
void GBlockProductPooling::forwardProp(const GVec& input, GVec& output) const
{
	size_t j = 0;
	for(size_t i = 0; i + 1 < input.size(); i += 2)
		output[j++] = input[i] * input[i + 1];
}

void GBlockProductPooling::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	for(size_t i = 0; i < outputs(); i++)
	{
		inBlame[2 * i] += outBlame[i] * input[2 * i + 1];
		inBlame[2 * i + 1] += outBlame[i] * input[2 * i];
	}
}








GMaxPooling2D::GMaxPooling2D(size_t inputCols, size_t inputRows, size_t inputChannels, size_t regionSize)
: GBlockWeightless(),
m_inputCols(inputCols),
m_inputRows(inputRows),
m_inputChannels(inputChannels),
m_regionSize(regionSize)
{
	if(inputCols % regionSize != 0)
		throw Ex("inputCols is not a multiple of regionSize");
	if(inputRows % regionSize != 0)
		throw Ex("inputRows is not a multiple of regionSize");
}

GMaxPooling2D::GMaxPooling2D(GDomNode* pNode)
: m_inputCols((size_t)pNode->field("icol")->asInt()),
m_inputRows((size_t)pNode->field("irow")->asInt()),
m_inputChannels((size_t)pNode->field("ichan")->asInt()),
m_regionSize((size_t)pNode->field("size")->asInt())
{
}

GMaxPooling2D::~GMaxPooling2D()
{
}

// virtual
GDomNode* GMaxPooling2D::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "icol", pDoc->newInt(m_inputCols));
	pNode->addField(pDoc, "irow", pDoc->newInt(m_inputRows));
	pNode->addField(pDoc, "ichan", pDoc->newInt(m_inputChannels));
	pNode->addField(pDoc, "size", pDoc->newInt(m_regionSize));
	return pNode;
}

// virtual
std::string GMaxPooling2D::to_str() const
{
	std::ostringstream os;
	os << "[GMaxPooling2D:" << GClasses::to_str(inputs()) << "->" << GClasses::to_str(outputs()) << "]";
	return os.str();
}

// virtual
void GMaxPooling2D::resize(size_t inputSize, size_t outputSize)
{
	if(inputSize != m_inputCols * m_inputRows * m_inputChannels)
		throw Ex("Changing the size of GMaxPooling2D is not supported");
	if(outputSize != m_inputChannels * m_inputCols * m_inputRows / (m_regionSize * m_regionSize))
		throw Ex("Changing the size of GMaxPooling2D is not supported");
}

// virtual
void GMaxPooling2D::forwardProp(const GVec& input, GVec& output) const
{
	size_t actPos = 0;
	for(size_t yy = 0; yy < m_inputRows; yy += m_regionSize)
	{
		for(size_t xx = 0; xx < m_inputCols; xx += m_regionSize)
		{
			for(size_t c = 0; c < m_inputChannels; c++)
			{
				double m = -1e100;
				size_t yStep = m_inputCols * m_inputChannels;
				size_t yStart = yy * yStep;
				size_t yEnd = yStart + m_regionSize * yStep;
				for(size_t y = yStart; y < yEnd; y += yStep)
				{
					size_t xStart = yStart + xx * m_inputChannels + c;
					size_t xEnd = xStart + m_regionSize * m_inputChannels + c;
					for(size_t x = xStart; x < xEnd; x += m_inputChannels)
						m = std::max(m, input[x]);
				}
				output[actPos++] = m;
			}
		}
	}
}

// virtual
void GMaxPooling2D::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	size_t downPos = 0;
	for(size_t yy = 0; yy < m_inputRows; yy += m_regionSize)
	{
		for(size_t xx = 0; xx < m_inputCols; xx += m_regionSize)
		{
			for(size_t c = 0; c < m_inputChannels; c++)
			{
				double m = -1e100;
				size_t maxIndex = 0;
				size_t yStep = m_inputCols * m_inputChannels;
				size_t yStart = yy * yStep;
				size_t yEnd = yStart + m_regionSize * yStep;
				for(size_t y = yStart; y < yEnd; y += yStep)
				{
					size_t xStart = yStart + xx * m_inputChannels + c;
					size_t xEnd = xStart + m_regionSize * m_inputChannels + c;
					for(size_t x = xStart; x < xEnd; x += m_inputChannels)
					{
						if(input[x] > m)
						{
							m = input[x];
							maxIndex = x;
						}
						inBlame[x] = 0.0;
					}
				}
				inBlame[maxIndex] = outBlame[downPos++];
			}
		}
	}
}











GBlockActivation::GBlockActivation(size_t size)
: GBlockWeightless()
{
	resize(size, size);
}

GBlockActivation::GBlockActivation(GDomNode* pNode)
: GBlockWeightless(pNode), m_units(pNode->field("units")->asInt())
{}

GDomNode* GBlockActivation::serialize(GDom* pDoc) const
{
	GDomNode* pNode = baseDomNode(pDoc);
	pNode->addField(pDoc, "units", pDoc->newInt(m_units));
	return pNode;
}

/// Makes a string representation of this layer
std::string GBlockActivation::to_str() const
{
	std::ostringstream oss;
	oss << "[GBlockActivation: type=" << (size_t)type() << ", size=" << inputs() << "]";
	return oss.str();
}

void GBlockActivation::resize(size_t in, size_t out)
{
	if(in != out)
		throw Ex("GBlockActivation must have the same number of inputs as outputs.");
	m_units = out;
}

void GBlockActivation::forwardProp(const GVec& input, GVec& output) const
{
	for(size_t i = 0; i < input.size(); i++)
		output[i] = eval(input[i]);
}

void GBlockActivation::backProp(const GVec& input, const GVec& output, const GVec& outBlame, GVec& inBlame)
{
	for(size_t i = 0; i < inBlame.size(); i++)
		inBlame[i] += outBlame[i] * derivative(input[i], output[i]);
}






} // namespace GClasses
