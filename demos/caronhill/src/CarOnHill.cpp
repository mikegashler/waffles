// -------------------------------------------------------------
// The contents of this file may be distributed under the CC0
// license (http://creativecommons.org/publicdomain/zero/1.0/).
// Of course you may also distribute it under a more restrictive
// license, such as any of the OSI-approved licenses
// (http://www.opensource.org/licenses).
// -------------------------------------------------------------

#include "CarOnHill.h"
#ifdef WINDOWS
#else
#	include <unistd.h>
#endif
#include <GClasses/GTime.h>
#include <GClasses/GThread.h>
#include <GClasses/GError.h>
#include <GClasses/GFile.h>
#include <GClasses/GRand.h>
#include <GClasses/GKNN.h>
#include <GClasses/GReinforcement.h>
#include <math.h>
#include <sstream>

using namespace GClasses;
using std::string;

bool g_updateView = true;

class CarQAgent : public GIncrementalLearnerQAgent
{
protected:
	double m_lastReward;
	GIncrementalLearner* m_pQTable;

public:
	CarQAgent(sp_relation& pRelation, double* pInitialState, GRand* prng, GAgentActionIterator* pActionIterator)
	: GIncrementalLearnerQAgent(pRelation, MakeQTable(prng), 1/*actionDims*/, pInitialState, prng, pActionIterator, 0.97/*softMaxThresh*/)
	{
		m_lastReward = UNKNOWN_REAL_VALUE;
	}

	virtual ~CarQAgent()
	{
		delete(m_pQTable);
	}

	GIncrementalLearner* MakeQTable(GRand* pRand)
	{
		size_t dims[3];
		dims[0] = 50; // position
		dims[1] = 50; // velocity
		dims[2] = 2; // actions
		m_pQTable = new GInstanceTable(3, dims, *pRand);
		return m_pQTable;
	}

	void SetLastReward(double d)
	{
		m_lastReward = d;
	}

	virtual double rewardFromLastAction()
	{
		return m_lastReward;
	}
};



class CarOnHillModel
{
protected:
	GImage* m_pImage;
	GImage* m_pCar;
	GImage* m_pRotatedCar;
	double m_carPos;
	double m_velocity;
	GDiscreteActionIterator* m_pActionIterator;
	GPolicyLearner* m_pAgents[1];
	GRand* m_prng;
	GWidgetTextLabel* m_pWins;
	int m_wins;

public:
	CarOnHillModel(GRand* prng, GImage* pImage, GWidgetTextLabel* pWins)
	{
		m_pWins = pWins;
		m_wins = 0;
		m_pImage = pImage;
		m_carPos = 0;
		m_velocity = 0;
		m_prng = prng;

		// Load the car image and add some border so we can rotate it
		GImage tmp;
		tmp.loadPng("minicar.png");
		m_pCar = new GImage();
		m_pCar->setSize(70, 60);
		GRect r(0, 0, 60, 36);
		m_pCar->blit(5, 5, &tmp, &r);
		m_pRotatedCar = new GImage();

		// Make the agent
		GMixedRelation* pRelAgent = new GMixedRelation();
		sp_relation relAgent;
		relAgent = pRelAgent;
		pRelAgent->addAttr(0); // position
		pRelAgent->addAttr(0); // velocity
		pRelAgent->addAttr(2); // action {forward, reverse}
		double initialState[2];
		initialState[0] = m_carPos;
		initialState[1] = m_velocity;
		double goalState[2];
		goalState[0] = 2;
		goalState[1] = 0;
		m_pActionIterator = new GDiscreteActionIterator(2);
		m_pAgents[0] = new CarQAgent(relAgent, initialState, m_prng, m_pActionIterator);
		((GQLearner*)m_pAgents[0])->setLearningRate(.9);
		((GQLearner*)m_pAgents[0])->setDiscountFactor(0.999);
	}

	virtual ~CarOnHillModel()
	{
		delete(m_pCar);
		delete(m_pRotatedCar);
		delete(m_pActionIterator);
		int i;
		for(i = 0; i < 1; i++)
			delete(m_pAgents[i]);
	}

	void DrawCar()
	{
		double angle = atan(3 * (2 * m_carPos - 3 * m_carPos * m_carPos));
		m_pRotatedCar->rotate(m_pCar, 35, 41, angle);
		double y = m_carPos * m_carPos - m_carPos * m_carPos * m_carPos;
		int i = (int)((m_carPos + 0.4) * m_pImage->width() / 1.4);
		int j = (int)(m_pImage->height() - y * 3 * m_pImage->height() - 50);
		GRect r(0, 0, 70, 60);
		m_pImage->blitAlpha(i - 35, j - 40, m_pRotatedCar, &r);
	}

	void Redraw(bool forw)
	{
		m_pImage->clear(0xff80c0e0);

		// Draw the hill
		{
			int i, j;
			double x, y;
			for(i = 0; i < (int)m_pImage->width(); i++)
			{
				x = (double)i * 1.4 / m_pImage->width() - 0.4;
				y = x * x - x * x * x;
				j = (int)(m_pImage->height() - y * 3 * m_pImage->height() - 50);
				m_pImage->lineNoChecks(i, j, i, m_pImage->height() - 1, 0xff40a060);
			}
		}

		// Draw the car
		DrawCar();

		// Draw the acceleration arrow
		m_pImage->arrow(240, 20, 240 + (forw ? 15 : -15), 20, 0xff000000, 10);
	}

	double DoAction(bool forw)
	{
		// Accelerate
		if(forw)
			m_velocity += .15;
		else
			m_velocity -= .15;

		// Friction
		m_velocity *= 0.98;

		// Move the car
		m_carPos += m_velocity / 50;
		if(m_carPos < -.4)
		{
			m_carPos = -.4;
			m_velocity = 0;
		}
		else if(m_carPos > .8)
		{
			m_carPos = 0;
			m_velocity = 0;
			m_wins++;
			std::ostringstream os;
			os << "Wins: " << m_wins;
			string tmp = os.str();
			m_pWins->setText(tmp.c_str());
			return UNKNOWN_REAL_VALUE;
		}

		return exp(15 * m_carPos);
	}

	void IterateModel(int sel, bool forw)
	{
		double slope = 2 * m_carPos - 3 * m_carPos * m_carPos;
		m_velocity -= slope;

		if(sel > 0)
		{
			// Do agent controls
			double senses[2];
			senses[0] = (m_carPos + 1.0) * 15.0;
			senses[1] = (m_velocity + 7.0) * 4.0;
			double action;
			m_pAgents[sel - 1]->refinePolicyAndChooseNextAction(senses, &action);
			forw = (action > 0);
		}
		double reward = DoAction(forw);
		if(sel == 1)
			((CarQAgent*)m_pAgents[0])->SetLastReward(reward);
		if(g_updateView)
			Redraw(forw);
	}
};


class CarOnHillDialog : public GWidgetDialog
{
protected:
	CarOnHillController* m_pController;
	GWidgetCanvas* m_pCanvas;
	GImage* m_pImage;
	CarOnHillModel* m_pModel;
	GRand* m_prng;
	GWidgetBulletGroup* m_pBullets;
	GWidgetCheckBox* m_pUpdateDisplay;
	GWidgetTextLabel* m_pWins;
	bool m_bPrevUpdate;

public:
	CarOnHillDialog(CarOnHillController* pController, int w, int h)
	: GWidgetDialog(w, h, 0xff90d0f0)
	{
		m_pController = pController;

		m_pBullets = new GWidgetBulletGroup(this, 820, 102, 14, 14, 2, 30, true);
		new GWidgetTextLabel(this, 840, 100, 100, 24, "Mouse", 0xff306000);
		new GWidgetTextLabel(this, 840, 130, 100, 24, "Q-Learner", 0xff306000);
		m_pBullets->setSelection(1);

		m_pWins = new GWidgetTextLabel(this, 820, 300, 100, 24, "Wins: 0", 0xff306000, 0xff90d0f0);

		m_pUpdateDisplay = new GWidgetCheckBox(this, 820, 402, 18, 18);
		m_pUpdateDisplay->setChecked(true);
		new GWidgetTextLabel(this, 840, 400, 100, 24, "Slow", 0xff306000);

		m_pImage = new GImage();
		m_pImage->setSize(800, 600);

		m_pCanvas = new GWidgetCanvas(this, 10, 30, m_pImage->width(), m_pImage->height(), m_pImage);
		m_prng = new GRand(0);
		m_pModel = new CarOnHillModel(m_prng, m_pImage, m_pWins);
		m_bPrevUpdate = true;
	}

	virtual ~CarOnHillDialog()
	{
		delete(m_pImage);
		delete(m_pModel);
		delete(m_prng);
	}

	void Iterate(bool forw)
	{
		g_updateView = m_pUpdateDisplay->isChecked();
		if(!g_updateView && m_prng->next(8192) == 0)
			g_updateView = true;
		m_pModel->IterateModel(m_pBullets->selection(), forw);
		m_pCanvas->setImage(m_pImage);
	}
};



// -------------------------------------------------------------------------------

class CarOnHillView : public ViewBase
{
protected:
	CarOnHillDialog* m_pDialog;
	bool m_forw;

public:
	CarOnHillView(CarOnHillController* pController)
	: ViewBase()
	{
		m_pDialog = new CarOnHillDialog(pController, m_screenRect.w, m_screenRect.h);
	}

	virtual ~CarOnHillView()
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
		if(x > 250)
			m_forw = true;
		else
			m_forw = false;
		return m_pDialog->handleMousePos(x - m_screenRect.x, y - m_screenRect.y);
	}

	void iterate()
	{
		m_pDialog->Iterate(m_forw);
	}

protected:
	virtual void draw(SDL_Surface *pScreen)
	{
		m_pDialog->Iterate(m_forw);

		// Clear the screen
		SDL_FillRect(pScreen, NULL/*&r*/, 0x000000);

		// Draw the dialog
		blitImage(pScreen, m_screenRect.x, m_screenRect.y, m_pDialog->image());
	}
};


// -------------------------------------------------------------------------------


CarOnHillController::CarOnHillController()
: ControllerBase()
{
	m_pView = new CarOnHillView(this);
}

CarOnHillController::~CarOnHillController()
{
	delete(m_pView);
}

void CarOnHillController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		if(handleEvents(time - timeOld) || g_updateView)
			m_pView->update();
		else
			((CarOnHillView*)m_pView)->iterate();
		timeOld = time;
	}
}

