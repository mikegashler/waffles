// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <GClasses/GActivation.h>
#include <GClasses/GApp.h>
#include <GClasses/GCluster.h>
#include <GClasses/GDecisionTree.h>
#include <GClasses/GEnsemble.h>
#include <GClasses/GTransform.h>
#include <GClasses/GKNN.h>
#include <GClasses/GLearner.h>
#include <GClasses/GLinear.h>
#include <GClasses/GError.h>
#include <GClasses/GNaiveBayes.h>
#include <GClasses/GNaiveInstance.h>
#include <GClasses/GNeuralNet.h>
#include <GClasses/GRand.h>
#include <GClasses/GSpinLock.h>
#include <GClasses/GTime.h>
#include <GClasses/GVec.h>
#include <GClasses/GWidgets.h>
#include <GClasses/GThread.h>
#include "Gui.h"
#include <exception>
#include <iostream>
#ifdef WINDOWS
#	include <direct.h>
#endif

using namespace GClasses;
using std::cout;
using std::cerr;

class InterpolateController : public ControllerBase
{
protected:

public:
	InterpolateController();
	virtual ~InterpolateController();

	void RunModal();
};


#define USE_IMAGE

class InterpolateDialog : public GWidgetDialog
{
protected:
	InterpolateController* m_pController;
	GImage* m_pImageSmall;
	GImage* m_pImageBig;
	size_t m_scale;
	GWidgetCanvas* m_pCanvasSmall;
	GWidgetCanvas* m_pCanvasBig;
	GWidgetGrid* m_pModels;
	GRand* m_pRand;
	bool m_animate;
	GMatrix* m_pTrainingFeatures;
	GMatrix* m_pTrainingLabels;
	GNeuralNet* m_pNN;
	GNeuralNet* m_pNN2;
	GSpinLock m_weightsLock;
	size_t m_workerMode; // 0 = exit, 1 = hybernate, 2 = yield, 3 = train

public:
	InterpolateDialog(InterpolateController* pController, int w, int h)
	: GWidgetDialog(w, h, 0xff90d0f0) // a=ff, r=90, g=d0, b=f0
	{
		m_pController = pController;
		m_pRand = new GRand(0);
		m_pImageSmall = new GImage();
#ifdef USE_IMAGE
		m_pImageSmall->loadPng("input.png");
#else
		m_pImageSmall->setSize(50, 50);
		m_pImageSmall->clear(0xffff8080);
		m_pImageSmall->circleFill(25, 25, 10.0, 0xff80ff80);
#endif
		//m_pImageSmall->loadPng("input.png");
		m_scale = 12;
		m_pImageBig = new GImage();
		m_pImageBig->setSize(m_pImageSmall->width() * m_scale, m_pImageSmall->height() * m_scale);
		m_pCanvasSmall = new GWidgetCanvas(this, 5, 35, m_pImageSmall->width(), m_pImageSmall->height(), m_pImageSmall);
		m_pCanvasBig = new GWidgetCanvas(this, 310, 5, m_pImageBig->width(), m_pImageBig->height(), m_pImageBig);
		m_pModels = new GWidgetGrid(this, 2, 5, 90, 290, 500, 0xff60a0c0);
		m_pModels->setRowHeight(36);
		m_pModels->setColumnWidth(0, 30);
		m_pModels->setColumnWidth(1, 242);
		m_pNN = NULL;
		m_pNN2 = NULL;

		// Make the training data
		m_pTrainingFeatures = new GMatrix(m_pImageSmall->width() * m_pImageSmall->height(), 2);
		m_pTrainingLabels = new GMatrix(m_pImageSmall->width() * m_pImageSmall->height(), 3);
		size_t ranges[2];
		ranges[0] = m_pImageSmall->width();
		ranges[1] = m_pImageSmall->height();
		GCoordVectorIterator cvi(2, ranges);
		unsigned int* pPix = m_pImageSmall->pixels();
		for(size_t i = 0; true; i++)
		{
			GAssert(i < m_pTrainingFeatures->rows());
			double* pFeature = m_pTrainingFeatures->row(i);
			cvi.currentNormalized(pFeature);
			GAssert(i < m_pTrainingLabels->rows());
			double* pLabel = m_pTrainingLabels->row(i);
			pLabel[0] = ((double)gRed(*pPix)) / 255;
			pLabel[1] = ((double)gGreen(*pPix)) / 255;
			pLabel[2] = ((double)gBlue(*pPix)) / 255;
			if(!cvi.advance())
				break;
			pPix++;
		}

		// Launch the worker thread
		m_workerMode = 1; // hybernate
		/*HANDLE threadHandle = */GThread::spawnThread(launchworkerThread, this);

		addModels();
	}

	virtual ~InterpolateDialog()
	{
		m_workerMode = 0; // tell the worker thread to exit
		GThread::sleep(2000); // todo: implement a better way to wait for the worker thread to exit
		delete(m_pTrainingFeatures);
		delete(m_pTrainingLabels);
		delete(m_pImageSmall);
		delete(m_pImageBig);
		delete(m_pRand);
		delete(m_pNN);
		delete(m_pNN2);
	}

	virtual void onCheckBulletHole(GWidgetBulletHole* pBullet)
	{
		int selectedIndex = -1;
		int n = m_pModels->rowCount();
		for(int i = 0; i < n; i++)
		{
			GWidgetBulletHole* pBH = (GWidgetBulletHole*)m_pModels->widget(0, i);
			if(pBH)
			{
				if(pBH == pBullet)
					selectedIndex = i;
				else
					pBH->setChecked(false);
			}
		}
		if(selectedIndex >= 0)
			onChoice(selectedIndex);
	}

	void addModel(const char* descr)
	{
		int index = m_pModels->rowCount();
		m_pModels->setWidget(0, index, new GWidgetBulletHole(m_pModels, 0, 0, 20, 20));
		GWidgetTextLabel* pDescription = new GWidgetTextLabel(m_pModels, 0, 0, 242, 36, descr);
		pDescription->wrap();
		m_pModels->setWidget(1, index, pDescription);
	}

	void doModel(GTransducer* pModel)
	{
		// Make the unlabeled data
		GMatrix dataUnlabeled(m_pImageBig->width() * m_pImageBig->height(), 2);
		size_t ranges[2];
		ranges[0] = m_pImageBig->width();
		ranges[1] = m_pImageBig->height();
		GCoordVectorIterator cvi(2, ranges);
		for(size_t i = 0; true; i++)
		{
			double* pFeature = dataUnlabeled.row(i);
			cvi.currentNormalized(pFeature);
			if(!cvi.advance())
				break;
		}

		// Predict labels
		GMatrix* pLabels = pModel->transduce(*m_pTrainingFeatures, *m_pTrainingLabels, dataUnlabeled);
		Holder<GMatrix> hLabels(pLabels);

		// Copy the labels into the big image
		cvi.reset();
		unsigned int* pPix = m_pImageBig->pixels();
		for(size_t i = 0; true; i++)
		{
			double* pRow = pLabels->row(i);
			int r = ClipChan((int)(pRow[0] * 255.0));
			int g = ClipChan((int)(pRow[1] * 255.0));
			int b = ClipChan((int)(pRow[2] * 255.0));
			*pPix = gARGB(0xff, r, g, b);
			if(!cvi.advance())
				break;
			pPix++;
		}
		m_pCanvasBig->setImage(m_pImageBig);
	}

	// takes ownership of pNN
	void doBackProp(GNeuralNet* pNN)
	{
		delete(m_pNN);
		m_pNN = pNN;
		sp_relation pFeatureRel = new GUniformRelation(2);
		sp_relation pLabelRel = new GUniformRelation(3);
		m_pNN->enableIncrementalLearning(pFeatureRel, pLabelRel);
		delete(m_pNN2);
		m_pNN2 = new GNeuralNet(m_pRand);
		m_pNN2->copyStructure(m_pNN);
		m_workerMode = 3; // train
	}

	void addModels()
	{
		addModel("1-nn"); // 0
		addModel("2-nn, equal weighting"); // 1
		addModel("2-nn, linear weighting"); // 2
		addModel("4-nn, equal weighting"); // 3
		addModel("4-nn, linear weighting"); // 4
		addModel("8-nn, equal weighting"); // 5
		addModel("8-nn, linear weighting"); // 6
		addModel("Naive Bayes"); // 7
		addModel("Decision tree"); // 8
		addModel("MeanMargins tree"); // 9
		addModel("Bag of 30 Decision trees"); // 10
		addModel("Bag of 30 MeanMargins trees"); // 11
		addModel("Perceptron"); // 12
		addModel("Neural Net, 2-15-60-3, logistic, online back-prop"); // 13
		addModel("Naive Instance 20"); // 14
		addModel("Mean label (a.k.a. baseline)"); // 15
		addModel("NeuralNet 2-16-60-3, gaussian activation"); // 16
	}

	void onChoice(int index)
	{
		m_workerMode = 1;
		if(index == 0)
		{
			GKNN model(1, m_pRand);
			doModel(&model);
		}
		else if(index == 1)
		{
			GKNN model(2, m_pRand);
			model.setInterpolationMethod(GKNN::Mean);
			doModel(&model);
		}
		else if(index == 2)
		{
			GKNN model(2, m_pRand);
			model.setInterpolationMethod(GKNN::Linear);
			doModel(&model);
		}
		else if(index == 3)
		{
			GKNN model(4, m_pRand);
			model.setInterpolationMethod(GKNN::Mean);
			doModel(&model);
		}
		else if(index == 4)
		{
			GKNN model(4, m_pRand);
			model.setInterpolationMethod(GKNN::Linear);
			doModel(&model);
		}
		else if(index == 5)
		{
			GKNN model(8, m_pRand);
			model.setInterpolationMethod(GKNN::Mean);
			doModel(&model);
		}
		else if(index == 6)
		{
			GKNN model(8, m_pRand);
			model.setInterpolationMethod(GKNN::Linear);
			doModel(&model);
		}
		else if(index == 7)
		{
			GNaiveBayes model(m_pRand);
			doModel(&model);
		}
		else if(index == 8)
		{
			GDecisionTree model(m_pRand);
			doModel(&model);
		}
		else if(index == 9)
		{
			GMeanMarginsTree model(m_pRand);
			doModel(&model);
		}
		else if(index == 10)
		{
			GBag bag(m_pRand);
			for(int i = 0; i < 30; i++)
			{
				GDecisionTree* pTree = new GDecisionTree(m_pRand);
				pTree->useRandomDivisions();
				bag.addLearner(pTree);
			}
			doModel(&bag);
		}
		else if(index == 11)
		{
			GBag bag(m_pRand);
			for(int i = 0; i < 30; i++)
			{
				GMeanMarginsTree* pTree = new GMeanMarginsTree(m_pRand);
				bag.addLearner(pTree);
			}
			doModel(&bag);
		}
		else if(index == 12)
		{
			GNeuralNet model(m_pRand);
			model.setActivationFunction(new GActivationIdentity(), true);
			doModel(&model);
		}
		else if(index == 13)
		{
			GNeuralNet* pNN = new GNeuralNet(m_pRand);
			pNN->addLayer(15);
			pNN->addLayer(60);
			doBackProp(pNN);
		}
		else if(index == 14)
		{
			GNaiveInstance model(20);
			doModel(&model);
		}
		else if(index == 15)
		{
			GBaselineLearner model;
			doModel(&model);
		}
		else if(index == 16)
		{
			GNeuralNet* pNN = new GNeuralNet(m_pRand);
//			pNN->setActivationFunction(new GActivationPiecewise(), true);
//			pNN->setActivationFunction(new GActivationSinc(), true);
			pNN->setActivationFunction(new GActivationGaussian(), true);
			pNN->addLayer(15);
			pNN->addLayer(60);
			doBackProp(pNN);
		}
	}

	void workerThread()
	{
		while(m_workerMode > 0)
		{
			if(m_workerMode == 3) // train
			{
				GSpinLockHolder hLock(&m_weightsLock, "training the network");
				for(size_t i = 0; i < 100; i++)
				{
					size_t r = (size_t)m_pRand->next(m_pTrainingFeatures->rows());
					m_pNN->trainIncremental(m_pTrainingFeatures->row(r), m_pTrainingLabels->row(r));
				}
			}
			else if(m_workerMode == 2) // yield
				GThread::sleep(0);
			else
				GThread::sleep(200); // hybernate
		}
	}

	static unsigned int launchworkerThread(void* pThis)
	{
		((InterpolateDialog*)pThis)->workerThread();
		return 0;
	}

	void Iterate()
	{
		if(m_workerMode == 3) // train
		{
			// Copy the weights
			m_workerMode = 2; // yield
			{
				GSpinLockHolder hLock(&m_weightsLock, "updating display");
				m_pNN2->copyWeights(m_pNN);
			}
			m_workerMode = 3; // train

			// Display the image
			GTEMPBUF(double, row, 5);
			size_t ranges[2];
			ranges[0] = m_pImageBig->width();
			ranges[1] = m_pImageBig->height();
			GCoordVectorIterator cvi(2, ranges);
			unsigned int* pPix = m_pImageBig->pixels();
			for(size_t i = 0; true; i++)
			{
				cvi.currentNormalized(row);
				m_pNN2->predict(row, row + 2);
				int r = ClipChan((int)(row[2] * 255.0));
				int g = ClipChan((int)(row[3] * 255.0));
				int b = ClipChan((int)(row[4] * 255.0));
				*pPix = gARGB(0xff, r, g, b);			
				if(!cvi.advance())
					break;
				pPix++;
			}
			m_pCanvasBig->setImage(m_pImageBig);

			// Slow down the display thread a little bit
			GThread::sleep(100);
		}
	}
};


// -------------------------------------------------------------------------------

class InterpolateView : public ViewBase
{
protected:
	InterpolateDialog* m_pDialog;

public:
	InterpolateView(InterpolateController* pController)
	: ViewBase()
	{
		m_pDialog = new InterpolateDialog(pController, m_screenRect.w, m_screenRect.h);
	}

	virtual ~InterpolateView()
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
		m_pDialog->Iterate();

		// Clear the screen
		SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);

		// Draw the dialog
		blitImage(pScreen, m_screenRect.x, m_screenRect.y, m_pDialog->image());
	}
};


// -------------------------------------------------------------------------------


InterpolateController::InterpolateController()
: ControllerBase()
{
	m_pView = new InterpolateView(this);
}

InterpolateController::~InterpolateController()
{
	delete(m_pView);
}

void InterpolateController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		handleEvents(time - timeOld);
		m_pView->update();
		GThread::sleep(50); // slow things down
		timeOld = time;
	}
}





int main(int argc, char *argv[])
{
#ifdef WINDOWS
	if(_chdir("../bin") != 0)
#else
	if(chdir("../bin") != 0)
#endif
	{
	}
	int nRet = 0;
	try
	{
		InterpolateController c;
		c.RunModal();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

