// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GApp.h>
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
	GMatrix m_tar;
	GMatrix m_context;
	GMatrix m_pred;
	GRand m_rand;
	GNeuralNet m_nn;
	size_t* m_pIndexes;
	size_t m_nextSpotlight;
	double m_errorThresh;
	double m_targetActive;
	size_t m_epochs;

public:
	UBPModel()
	: m_tar(0, 5), m_context(0, 2), m_pred(0, 5), m_rand(0), m_nn(m_rand), m_errorThresh(0.0), m_targetActive(1.0)
	{
		{
			GMatrix* pTmp = GMatrix::loadArff("in.arff");
			Holder<GMatrix> hTmp(pTmp);
			if(pTmp->cols() != 3)
				ThrowError("unexpected number of columns");
			m_tar.newRows(pTmp->rows());
			m_tar.copyColumns(0, pTmp, 0, 3);
		}
		m_pred.newRows(m_tar.rows());
		m_context.newRows(m_tar.rows());
		for(size_t i = 0; i < m_tar.rows(); i++)
		{
			double* pContext = m_context.row(i);
			//pContext[0] = 0.1 * m_rand.normal();
			//pContext[1] = 0.1 * m_rand.normal();
			GVec::setAll(pContext, 0.5, 2);
			//m_rand.cubical(pContext, 2);
			double* pTar = m_tar.row(i);
			pTar[3] = 0.8 * i / m_tar.rows();
			pTar[4] = 3.0;
			double* pPred = m_pred.row(i);
			pPred[3] = 0.8 * i / m_pred.rows();
			pPred[4] = 10.0;
		}
		m_nn.addLayer(36);
		m_nn.setLearningRate(0.1);
		sp_relation pFeatureRel = new GUniformRelation(2);
		sp_relation pLabelRel = new GUniformRelation(3);
		m_nn.beginIncrementalLearning(pFeatureRel, pLabelRel);
		m_pIndexes = new size_t[m_tar.rows()];
		GIndexVec::makeIndexVec(m_pIndexes, m_tar.rows());
		m_nextSpotlight = 0;
	}

	~UBPModel()
	{
		m_context.saveArff("out.arff");
		delete[] m_pIndexes;
	}

	GNeuralNet& nn() { return m_nn; }

	void update()
	{
		double* pSpotlightTarget = m_tar.row(m_nextSpotlight);
		double* pSpotlightContext = m_context.row(m_nextSpotlight);
		m_nn.forwardProp(pSpotlightContext);
		double spotlightErr = m_nn.sumSquaredPredictionError(pSpotlightTarget);
		double maxBelowThresh = 0.0;
		double minAboveThresh = 1e308;
		double cumErr = 0.0;
		size_t activeCount = 0;
		GBackProp* pBP = m_nn.backProp();
		GBackPropLayer& bpLayer = pBP->layer(m_nn.layerCount() - 1);
		GIndexVec::shuffle(m_pIndexes, m_tar.rows(), &m_rand);
		size_t* pInd = m_pIndexes;
		for(size_t i = 0; i < m_tar.rows(); i++)
		{
			// Adjust the weights
			size_t index = *(pInd++);
			double* pTar = m_tar.row(index);
			double* pPred = m_pred.row(index);
			double* pContext = m_context.row(index);
			m_nn.forwardProp(pContext);
			m_nn.copyPrediction(pPred);
			m_nn.setErrorOnOutputLayer(pTar/*, GNeuralNet::root_cubed_error*/);
			
			//double err = m_nn.sumSquaredPredictionError(pTar);
			double err = 0.0;
			for(vector<GBackPropNeuron>::iterator it = bpLayer.m_neurons.begin(); it != bpLayer.m_neurons.end(); it++)
				err += it->m_error * it->m_error;

			if(err < m_errorThresh)
			{
				pPred[4] = 10.0; // big radius
				pBP->backpropagate();
				activeCount++;
				maxBelowThresh = std::max(maxBelowThresh, err);
				pBP->descendGradient(pContext, m_nn.learningRate(), 0.0, false);
				pBP->adjustFeatures(pContext, m_nn.learningRate(), 0, false);
//				GVec::capValues(pContext, 1.0, 2);
//				GVec::floorValues(pContext, 0.0, 2);

				// See if we can improve the spotlight point
				double sse = m_nn.sumSquaredPredictionError(pTar);
				cumErr += sse;
				if(m_rand.uniform() * cumErr < sse)
					m_nextSpotlight = index;
				double err2 = m_nn.sumSquaredPredictionError(pSpotlightTarget);
				if(err2 < spotlightErr)
				{
					spotlightErr = err2;
					GVec::copy(pSpotlightContext, pContext, 2);
				}
			}
			else
			{
				pPred[4] = 0.0; // no radius
				minAboveThresh = std::min(minAboveThresh, err);
				GVec::copy(pContext, m_context.row((size_t)m_rand.next(m_context.rows())), 2);
			}
		}
		if(activeCount <= (size_t)floor(m_targetActive))
			m_errorThresh = minAboveThresh + 1e-9;
		else
			m_errorThresh = maxBelowThresh;
		m_targetActive += 0.4;

		// Update the threshold
//		cout << "target active: " << m_targetActive << ", active: " << activeCount << ", thresh: " << m_errorThresh << "\n";
	}

	GMatrix& pred() { return m_pred; }

	GMatrix& tar() { return m_tar; }
	GMatrix& context() { return m_context; }
};

class Compare3DPointsByDistanceFromCameraFunctor
{
protected:
	GCamera* m_pCamera;

public:
	Compare3DPointsByDistanceFromCameraFunctor(GCamera* pCamera)
	: m_pCamera(pCamera)
	{
	}

	// returns false if pA is closer than pB
	bool operator() (const double* pA, const double* pB) const
	{
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
	GMatrix* m_pDataSorter;
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
		m_pDataSorter = NULL;
		m_pButtonYawN = new GWidgetTextButton(this, 600, 150, 50, 30, "-Yaw");
		m_pButtonYawP = new GWidgetTextButton(this, 700, 150, 50, 30, "+Yaw");
		m_pButtonPitN = new GWidgetTextButton(this, 650, 100, 50, 30, "+Pit");
		m_pButtonPitP = new GWidgetTextButton(this, 650, 200, 50, 30, "-Pit");
		new GWidgetTextLabel(this, 200, 5, 600, 40, "Unsupervised Back-Propagation", 0xff000000, 0, 3.0f);
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
		m_pDataSorter->releaseAllRows();
		delete(m_pDataSorter);
	}

	void makeCamera(GImage* pImage, GMatrix* pDataTar, GMatrix* pDataPred)
	{
		m_pCamera = new GCamera(pImage->width(), pImage->height());
		m_pCamera->setViewAngle(M_PI / 3);
		m_centroid.m_vals[0] = pDataTar->mean(0);
		m_centroid.m_vals[1] = pDataTar->mean(1);
		m_centroid.m_vals[2] = pDataTar->mean(2);
		G3DVector min, max, range;
		pDataTar->minAndRangeUnbiased(0, &min.m_vals[0], &range.m_vals[0]);
		pDataTar->minAndRangeUnbiased(1, &min.m_vals[1], &range.m_vals[1]);
		pDataTar->minAndRangeUnbiased(2, &min.m_vals[2], &range.m_vals[2]);
		max.copy(&range);
		max.add(&min);
		updateCameraDirection();

		m_pDataSorter = new GMatrix(pDataTar->relation());
		m_pDataSorter->reserve(pDataTar->rows() + pDataPred->rows());
		for(size_t i = 0; i < pDataTar->rows(); i++)
			m_pDataSorter->takeRow(pDataTar->row(i));
		for(size_t i = 0; i < pDataPred->rows(); i++)
			m_pDataSorter->takeRow(pDataPred->row(i));
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

	void makeImage(GImage* pImage, GMatrix* pDataTar, GMatrix* pDataPred)
	{
		m_pImage->clear(0xffffffff);
		G3DVector point, coords, point2, coords2;
		Compare3DPointsByDistanceFromCameraFunctor comparator(m_pCamera);
		m_pDataSorter->sort(comparator);
		for(size_t i = 0; i < m_pDataSorter->rows(); i++)
		{
			double* pVec = m_pDataSorter->row(i);
			point.set(pVec[0], pVec[1], pVec[2]);
			toImageCoords(pImage, m_pCamera, &point, &coords);
			float radius = (float)pVec[4] / (float)coords.m_vals[2];
			pImage->dot((float)coords.m_vals[0], (float)coords.m_vals[1], radius, gAHSV(0xff, (float)pVec[3], 1.0f, 0.5f), 0xffffffff);
		}
	}

	void makeImage2(GImage* pImage, GMatrix* pData, GMatrix* pDataPred)
	{
		pImage->clear(0xffffffff);
		GPlotWindow pw(pImage, m_viewRect.x, m_viewRect.y, m_viewRect.x + m_viewRect.w, m_viewRect.y + m_viewRect.h);
		for(size_t i = 0; i < pData->rows(); i++)
		{
			double* pVec = pData->row(i);
			double* pPred = pDataPred->row(i);
			pw.dot(pVec[0], pVec[1], 0.3f * (float)pPred[4], gAHSV(0xff, 0.8f * (float)i / pData->rows(), 1.0f, 0.5f), 0xffffffff);
			m_viewRect.include(pVec[0], pVec[1]);
		}
	}

	void update()
	{
		UBPModel* pModel = m_pController->model();
		GMatrix& tar = pModel->tar();
		GMatrix& pred = pModel->pred();
		if(!m_pCamera)
			makeCamera(m_pImage, &tar, &pred);
		makeImage(m_pImage, &tar, &pred);
		m_pCanvas->setImage(m_pImage);
		GMatrix& context = pModel->context();
		makeImage2(m_pImage2, &context, &pred);
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

