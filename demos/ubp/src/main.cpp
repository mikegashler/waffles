/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    anonymous contributors,

  to the public domain (http://creativecommons.org/publicdomain/zero/1.0/).

  Note that some moral obligations still exist in the absence of legal ones.
  For example, it would still be dishonest to deliberately misrepresent the
  origin of a work. Although we impose no legal requirements to obtain a
  license, it is beseeming for those who build on the works of others to
  give back useful improvements, or pay it forward in their own field. If
  you would like to cite us, a published paper about Waffles can be found
  at http://jmlr.org/papers/volume12/gashler11a/gashler11a.pdf. If you find
  our code to be useful, the Waffles team would love to hear how you use it.
*/

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GApp.h>
#include <GClasses/GActivation.h>
#include <GClasses/G3D.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GError.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GPlot.h>
#include <GClasses/GVec.h>
#include <GClasses/GRand.h>
#include <GClasses/GMath.h>
#include <GClasses/GTime.h>
#include <GClasses/GWidgets.h>
#include <GClasses/GThread.h>
#include "Gui.h"
#ifdef WINDOWS
#	include <direct.h>
#endif
#include <exception>
#include <iostream>
#include <vector>

using namespace GClasses;
using std::cerr;
using std::cout;
using std::vector;

class UBPModel;


class UBPController : public ControllerBase
{
protected:
	UBPModel* m_pModel;

public:
	UBPController();
	virtual ~UBPController();

	UBPModel* model() { return m_pModel; }
	void RunModal();
	void next();
};



class UBPModel
{
protected:
	GMatrix m_target;
	GMatrix m_intrinsic;
	GMatrix m_predicted;

	GRand m_rand;
	GNeuralNet m_nn;
	size_t* m_pIndexes;
	size_t m_nextSpotlight;
	double m_errorThresh;
	double m_targetActive;
	size_t m_epochs;
	GVec m_inputGradient;

public:
	UBPModel()
	: m_rand(0), m_errorThresh(0.0), m_targetActive(1.0), m_inputGradient(2)
	{
		m_target.loadArff("in.arff");
		if(m_target.cols() != 3)
			throw Ex("unexpected number of columns");
		m_predicted.resize(m_target.rows(), 3);
		m_intrinsic.resize(m_target.rows(), 2);
		for(size_t i = 0; i < m_intrinsic.rows(); i++)
		{
			GVec& pRow = m_intrinsic.row(i);
			pRow[0] = 0.5; //0.1 * m_rand.normal();
			pRow[1] = 0.5; //0.1 * m_rand.normal();
		}
		m_nn.addLayer(new GLayerClassic(FLEXIBLE_SIZE, 50));
		m_nn.addLayer(new GLayerClassic(50, FLEXIBLE_SIZE));
		m_nn.setLearningRate(0.1);
		GUniformRelation featureRel(2);
		GUniformRelation labelRel(3);
		m_nn.beginIncrementalLearning(featureRel, labelRel);
		m_pIndexes = new size_t[m_target.rows()];
		GIndexVec::makeIndexVec(m_pIndexes, m_target.rows());
		m_nextSpotlight = 0;
	}

	~UBPModel()
	{
		m_intrinsic.saveArff("out.arff");
		delete[] m_pIndexes;
	}

	GNeuralNet& nn() { return m_nn; }

	void update()
	{
		GVec& pSpotlightTarget = m_target.row(m_nextSpotlight);
		GVec& pSpotlightContext = m_intrinsic.row(m_nextSpotlight);
		m_nn.forwardProp(pSpotlightContext);
		double spotlightErr = m_nn.sumSquaredPredictionError(pSpotlightTarget);
		double maxBelowThresh = 0.0;
		double minAboveThresh = 1e308;
		double cumErr = 0.0;
		size_t activeCount = 0;
		GIndexVec::shuffle(m_pIndexes, m_target.rows(), &m_rand);
		size_t* pInd = m_pIndexes;
		for(size_t i = 0; i < m_target.rows(); i++)
		{
			// Adjust the weights
			size_t index = *(pInd++);
			GVec& pTar = m_target.row(index);
			GVec& pPred = m_predicted.row(index);
			GVec& pIntrinsic = m_intrinsic.row(index);
			m_nn.predict(pIntrinsic, pPred);
			m_nn.backpropagate(pTar);

			double err = m_nn.sumSquaredPredictionError(pTar);
			if(err < m_errorThresh)
			{
				activeCount++;
				maxBelowThresh = std::max(maxBelowThresh, err);
				m_nn.gradientOfInputs(m_inputGradient);
				m_nn.descendGradient(pIntrinsic, m_nn.learningRate(), 0.0);
				GVec::addScaled(pIntrinsic.data(), -m_nn.learningRate(), m_inputGradient.data(), 2);

				// See if we can improve the spotlight point
				double sse = m_nn.sumSquaredPredictionError(pTar);
				cumErr += sse;
				if(m_rand.uniform() * cumErr < sse)
					m_nextSpotlight = index;
				double err2 = m_nn.sumSquaredPredictionError(pSpotlightTarget);
				if(err2 < spotlightErr)
				{
					spotlightErr = err2;
					GVec::copy(pSpotlightContext.data(), pIntrinsic.data(), 2);
				}
			}
			else
			{
				minAboveThresh = std::min(minAboveThresh, err);
				GVec::copy(pIntrinsic.data(), m_intrinsic.row((size_t)m_rand.next(m_intrinsic.rows())).data(), 2);
			}
		}
		if(activeCount <= (size_t)floor(m_targetActive))
			m_errorThresh = minAboveThresh + 1e-9;
		else
			m_errorThresh = maxBelowThresh;
		m_targetActive += 0.2;

		// Update the threshold
//		cout << "target active: " << m_targetActive << ", active: " << activeCount << ", thresh: " << m_errorThresh << "\n";
	}

	GMatrix& pred() { return m_predicted; }
	GMatrix& tar() { return m_target; }
	GMatrix& intrinsic() { return m_intrinsic; }
};

class Compare3DPointsByDistanceFromCameraFunctor
{
protected:
	GCamera* m_pCamera;
	const GMatrix& m_target;
	const GMatrix& m_predicted;

public:
	Compare3DPointsByDistanceFromCameraFunctor(GCamera* pCamera, const GMatrix& target, const GMatrix& predicted)
	: m_pCamera(pCamera), m_target(target), m_predicted(predicted)
	{
	}

	// returns false if pA is closer than pB
	bool operator() (size_t aa, size_t bb) const
	{
		const GVec& pA = aa < 1000000 ? m_target[aa] : m_predicted[aa - 1000000];
		const GVec& pB = bb < 1000000 ? m_target[bb] : m_predicted[bb - 1000000];
		G3DVector a, b, c, d;
		a.m_vals[0] = pA[0];
		a.m_vals[1] = pA[1];
		a.m_vals[2] = pA[2];
		b.m_vals[0] = pB[0];
		b.m_vals[1] = pB[1];
		b.m_vals[2] = pB[2];
		m_pCamera->project(&a, &c);
		m_pCamera->project(&b, &d);
		return (c.m_vals[2] > d.m_vals[2]);
	}
};

void toImageCoords(GImage* pImage, GCamera* pCamera, G3DVector* pIn, G3DVector* pOut)
{
	pCamera->project(pIn, pOut);

	// Flip the Y value, because positive is down in image coordinates
	pOut->m_vals[1] = pImage->height() - 1 - pOut->m_vals[1];
}

class UBPDialog : public GWidgetDialog
{
protected:
	UBPController* m_pController;
	GImage* m_pImage;
	GImage* m_pImage2;
	GWidgetCanvas* m_pCanvas;
	GWidgetCanvas* m_pCanvas2;
	GCamera* m_pCamera;
	vector<size_t> m_indexesSortedByCameraDistance;
	G3DReal m_cameraDist;
	G3DVector m_cameraDirection;
	G3DVector m_centroid;
	double m_yaw, m_pitch;
	GWidgetTextButton* m_pButtonYawN;
	GWidgetTextButton* m_pButtonYawP;
	GWidgetTextButton* m_pButtonPitN;
	GWidgetTextButton* m_pButtonPitP;
	GDoubleRect m_viewRect;

public:
	UBPDialog(UBPController* pController, int w, int h)
	: GWidgetDialog(w, h, 0xff90d0f0), m_cameraDist(1.6), m_viewRect(0.0, 0.0, 1.0, 1.0)
	{
		m_pController = pController;
		m_pImage = new GImage();
		m_pImage->setSize(550, 550);
		m_pImage2 = new GImage();
		m_pImage2->setSize(350, 350);
		m_pCanvas = new GWidgetCanvas(this, 20, 100, m_pImage->width(), m_pImage->height(), m_pImage);
		m_pCanvas2 = new GWidgetCanvas(this, 620, 300, m_pImage2->width(), m_pImage2->height(), m_pImage2);
		m_pCamera = NULL;
		m_pButtonYawN = new GWidgetTextButton(this, 600, 150, 50, 30, "-Yaw");
		m_pButtonYawP = new GWidgetTextButton(this, 700, 150, 50, 30, "+Yaw");
		m_pButtonPitN = new GWidgetTextButton(this, 650, 100, 50, 30, "+Pit");
		m_pButtonPitP = new GWidgetTextButton(this, 650, 200, 50, 30, "-Pit");
		new GWidgetTextLabel(this, 200, 45, 600, 16, "(In this case, the model contains two inputs, one hidden layer with 36 units, and 3 outputs.)", 0xff000000, 0, 1.0f);
		new GWidgetTextLabel(this, 20, 80, 600, 24, "Training vectors are shown with small dots. Predicted output vectors are shown with big dots.", 0xff000000, 0, 1.0f);
		new GWidgetTextLabel(this, 620, 276, 300, 24, "Latent input vectors (not given in the training data).", 0xff000000, 0, 1.0f);
		m_yaw = 0.0;
		m_pitch = 1.0;
		moveCam(0.0, 0.0);
	}

	virtual ~UBPDialog()
	{
		delete(m_pImage);
		delete(m_pImage2);
		delete(m_pCamera);
	}

	void makeCamera(GImage* pImage, GMatrix* pDataTar, GMatrix* pDataPred)
	{
		m_pCamera = new GCamera(pImage->width(), pImage->height());
		m_pCamera->setViewAngle(M_PI / 3);
		m_centroid.m_vals[0] = pDataTar->columnMean(0);
		m_centroid.m_vals[1] = pDataTar->columnMean(1);
		m_centroid.m_vals[2] = pDataTar->columnMean(2);
		G3DVector min, max, range;
		min.m_vals[0] = pDataTar->columnMin(0);
		range.m_vals[0] = pDataTar->columnMax(0) - min.m_vals[0];
		min.m_vals[1] = pDataTar->columnMin(1);
		range.m_vals[1] = pDataTar->columnMax(1) - min.m_vals[1];
		min.m_vals[2] = pDataTar->columnMin(2);
		range.m_vals[2] = pDataTar->columnMax(2) - min.m_vals[2];
		max.copy(&range);
		max.add(&min);
		updateCameraDirection();

		m_indexesSortedByCameraDistance.clear();
		m_indexesSortedByCameraDistance.reserve(pDataTar->rows() + pDataPred->rows());
		for(size_t i = 0; i < pDataTar->rows(); i++)
			m_indexesSortedByCameraDistance.push_back(i);
		for(size_t i = 0; i < pDataPred->rows(); i++)
			m_indexesSortedByCameraDistance.push_back(1000000 + i);
	}

	void updateCameraDirection()
	{
		G3DReal dist = m_cameraDist;
		G3DVector* pCameraPos = m_pCamera->lookFromPoint();
		pCameraPos->copy(&m_cameraDirection);
		pCameraPos->multiply(-1);
		pCameraPos->normalize();
		pCameraPos->multiply(dist);
		pCameraPos->add(&m_centroid);
		m_pCamera->setDirection(&m_cameraDirection, 0.0/*roll*/);
	}

	void update()
	{
		UBPModel* pModel = m_pController->model();
		GMatrix& tar = pModel->tar();
		GMatrix& pred = pModel->pred();
		if(!m_pCamera)
			makeCamera(m_pImage, &tar, &pred);
		m_pImage->clear(0xffffffff);
		G3DVector point, coords, point2, coords2;
		Compare3DPointsByDistanceFromCameraFunctor comparator(m_pCamera, tar, pred);
		
		std::sort(m_indexesSortedByCameraDistance.begin(), m_indexesSortedByCameraDistance.end(), comparator);
		for(size_t i = 0; i < m_indexesSortedByCameraDistance.size(); i++)
		{
			size_t index = m_indexesSortedByCameraDistance[i];
			if(index < 1000000)
			{
				// Draw target point
				GVec& pVec = tar[index];
				point.set(pVec[0], pVec[1], pVec[2]);
				toImageCoords(m_pImage, m_pCamera, &point, &coords);
				float radius = (float)3.0 / (float)coords.m_vals[2];
				m_pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], radius, gAHSV(0xff, (float)(0.8 * index / tar.rows()), 1.0f, 0.5f), 0xffffffff);
			}
			else
			{
				// Draw predicted point
				index -= 1000000;
				GVec& pVec = pred[index];
				point.set(pVec[0], pVec[1], pVec[2]);
				toImageCoords(m_pImage, m_pCamera, &point, &coords);
				float radius = (float)6.0 / (float)coords.m_vals[2];
				m_pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], radius, gAHSV(0xff, (float)(0.8 * index / pred.rows()), 1.0f, 0.5f), 0xffffffff);
			}
		}
		m_pCanvas->setImage(m_pImage);
		GMatrix& intrinsic = pModel->intrinsic();

		// Make intrinsic image
		m_pImage2->clear(0xffffffff);
		GPlotWindow pw(m_pImage2, m_viewRect.x, m_viewRect.y, m_viewRect.x + m_viewRect.w, m_viewRect.y + m_viewRect.h);
		for(size_t i = 0; i < intrinsic.rows(); i++)
		{
			GVec& pVec = intrinsic.row(i);
			pw.dot(pVec[0], pVec[1], 2.5f, gAHSV(0xff, 0.8f * (float)i / intrinsic.rows(), 1.0f, 0.5f), 0xffffffff);
			m_viewRect.include(pVec[0], pVec[1]);
		}
		m_pCanvas2->setImage(m_pImage2);
	}

	void moveCam(double dYaw, double dPit)
	{
		m_yaw += dYaw;
		m_pitch = std::max(-M_PI / 2, std::min(M_PI / 2, m_pitch + dPit));
		m_cameraDirection.set(sin(m_yaw) * cos(m_pitch), sin(m_pitch), -cos(m_yaw) * cos(m_pitch));
		if(m_pCamera)
			updateCameraDirection();
	}

	virtual void onReleaseTextButton(GWidgetTextButton* pButton)
	{
		if(pButton == m_pButtonYawN)
			moveCam(-0.1, 0.0);
		else if(pButton == m_pButtonYawP)
			moveCam(0.1, 0.0);
		else if(pButton == m_pButtonPitN)
			moveCam(0.0, -0.1);
		else if(pButton == m_pButtonPitP)
			moveCam(0.0, 0.1);
		else
			GAssert(false); // unknown button
	}
};


// -------------------------------------------------------------------------------

class UBPView : public ViewBase
{
protected:
	UBPDialog* m_pDialog;

public:
	UBPView(UBPController* pController)
	: ViewBase()
	{
		m_pDialog = new UBPDialog(pController, m_screenRect.w, m_screenRect.h);
	}

	virtual ~UBPView()
	{
		delete(m_pDialog);
	}

	virtual void onChar(char c)
	{
		m_pDialog->handleChar(c);
	}

	virtual void onMouseDown(int nButton, int x, int y)
	{
		m_pDialog->pressButton(nButton, x - m_screenRect.x, y - m_screenRect.y);
	}

	virtual void onMouseUp(int nButton, int x, int y)
	{
		m_pDialog->releaseButton(nButton);
	}

	virtual bool onMousePos(int x, int y)
	{
		return m_pDialog->handleMousePos(x - m_screenRect.x, y - m_screenRect.y);
	}

protected:
	virtual void draw(SDL_Surface *pScreen)
	{
		m_pDialog->update();

		// Clear the screen
		SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);

		// Draw the dialog
		blitImage(pScreen, m_screenRect.x, m_screenRect.y, m_pDialog->image());
	}
};


// -------------------------------------------------------------------------------


UBPController::UBPController()
: ControllerBase()
{
	m_pModel = new UBPModel();
	m_pView = new UBPView(this);
}

UBPController::~UBPController()
{
	delete(m_pView);
	delete(m_pModel);
}

void UBPController::next()
{
	m_pModel->update();
}

void UBPController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		if(!handleEvents(time - timeOld))
			GThread::sleep(0); // slow things down
		m_pView->update();
		m_pModel->update();
		timeOld = time;
	}
}





int main(int argc, char *argv[])
{
	char szAppPath[300];
	GApp::appPath(szAppPath, 300, true);
#ifdef WINDOWS
	if(_chdir(szAppPath) != 0)
#else
	if(chdir(szAppPath) != 0)
#endif
	{}
	int nRet = 0;
	try
	{
		UBPController c;
		c.RunModal();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

