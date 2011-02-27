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
#include <GClasses/GError.h>
#include <GClasses/GRand.h>
#include <GClasses/GTime.h>
#include <GClasses/GWidgets.h>
#include <GClasses/GThread.h>
#include "Gui.h"
#include <GClasses/GChess.h>
#ifdef WINDOWS
#	include <direct.h>
#endif
#include <math.h>
#include <exception>
#include <iostream>

using namespace GClasses;
using std::cerr;

class ChessController : public ControllerBase
{
protected:

public:
	ChessController();
	virtual ~ChessController();

	void RunModal();
};


#define MAX_DEPTH_EVER 32

class ChessDialog : public GWidgetDialog
{
protected:
	GChessBoard m_board;
	ChessController* m_pController;
	GWidgetCanvas* m_pCanvas;
	GImage* m_pImage;
	GImage* m_pPieces;
	GRand* m_prng;
	int m_moves;
	int m_maxDepth;
	GChessMoveIterator* m_pMoveIterators;
	bool m_whitesTurn;
	GWidgetBulletGroup* m_pLightStrategy;
	GWidgetBulletGroup* m_pDarkStrategy;
	GChessBoard::Piece m_pieceInHand;
	int m_pieceInHandX;
	int m_pieceInHandY;
	bool m_pieceInHandWhite;
	int m_lastMove[4];
	int m_boardEvals;

	int m_winCounts[3];
	double m_lightEvals;
	double m_darkEvals;

	int m_hintMoves[MAX_PIECE_MOVES * 2];
	int m_hintMoveCount;

public:
	ChessDialog(ChessController* pController, int w, int h)
	: GWidgetDialog(w, h, 0xff90d0f0)
	{
		m_pController = pController;
		m_prng = new GRand(0);
		m_pImage = new GImage();
		m_pImage->setSize(560, 560);
		m_pImage->clear(0xff000000);
		m_pCanvas = new GWidgetCanvas(this, 10, 30, m_pImage->width(), m_pImage->height(), m_pImage);

		m_pLightStrategy = new GWidgetBulletGroup(this, 620, 68, 14, 14, 4, 30, true);
		new GWidgetTextLabel(this, 640, 70, 150, 20, "Human", 0xff306000);
		new GWidgetTextLabel(this, 640, 100, 150, 20, "Random", 0xff306000);
		new GWidgetTextLabel(this, 640, 130, 250, 20, "Alpha Beta Pruning (look ahead 3 moves)", 0xff306000);
		new GWidgetTextLabel(this, 640, 160, 250, 20, "Alpha Beta Pruning (look ahead 6 moves)", 0xff306000);
		m_pLightStrategy->setSelection(0);

		m_pDarkStrategy = new GWidgetBulletGroup(this, 620, 488, 14, 14, 4, 30, true);
		new GWidgetTextLabel(this, 640, 490, 150, 20, "Human", 0xff306000);
		new GWidgetTextLabel(this, 640, 520, 150, 20, "Random", 0xff306000);
		new GWidgetTextLabel(this, 640, 550, 250, 20, "Alpha Beta Pruning (look ahead 3 moves)", 0xff306000);
		new GWidgetTextLabel(this, 640, 580, 250, 20, "Alpha Beta Pruning (look ahead 6 moves)", 0xff306000);
		m_pDarkStrategy->setSelection(0);

		m_pPieces = new GImage();
		m_pPieces->loadPng("pieces.png");
		m_pPieces->replaceColor(0xff00ff00, 0x00000000); // Change opaque green to translucent black
		m_pMoveIterators = new GChessMoveIterator[MAX_DEPTH_EVER];
		m_maxDepth = 6;
		m_pieceInHand = GChessBoard::None;
		m_pieceInHandWhite = true;
		m_lastMove[0] = -1;
		m_lastMove[1] = -1;
		m_lastMove[2] = -1;
		m_lastMove[3] = -1;

		m_winCounts[0] = 0;
		m_winCounts[1] = 0;
		m_winCounts[2] = 0;

		RestartGame();
	}

	virtual ~ChessDialog()
	{
		delete(m_pImage);
		delete(m_prng);
		delete(m_pPieces);
		delete[] m_pMoveIterators;
	}

	void RestartGame()
	{
		printf("Light=%d, Dark=%d, Draw=%d le=%lg de=%lg\n", m_winCounts[0], m_winCounts[1], m_winCounts[2], m_lightEvals, m_darkEvals);
		m_moves = 0;
		m_whitesTurn = true;
		m_board.resetBoard();
		m_hintMoveCount = 0;
		m_lightEvals = 0;
		m_darkEvals = 0;
	}

	void SetLastMove(int xSrc, int ySrc, int xDest, int yDest)
	{
		m_lastMove[3] = m_lastMove[2];
		m_lastMove[2] = m_lastMove[1];
		m_lastMove[1] = m_lastMove[0];
		m_lastMove[0] = xSrc + 8 * (ySrc + 8 * (xDest + 8 * yDest));
	}

	// Returns true if this would be a repeat-agressive move (which is not allowed)
	bool IsRepeatAggressiveMove(int xSrc, int ySrc, int xDest, int yDest)
	{
		if(xDest + 8 * yDest == m_lastMove[3] / 64 && // this move takes the same square as my move-before-last, and
			(m_lastMove[2] & 63) == (m_lastMove[0] >> 6) && (m_lastMove[2] >> 6) == (m_lastMove[0] & 63) && // my opponent undid his last move, and
			m_lastMove[1] == xDest + 8 * (yDest + 8 * (xSrc + 8 * ySrc))) // this undoes my last move
			return true;
		return false;
	}

	void DrawBoard()
	{
		int x, y;
		bool white;
		GChessBoard::Piece piece;
		for(y = 0; y < 8; y++)
		{
			for(x = 0; x < 8; x++)
			{
				m_pImage->boxFill(70 * x, 70 * y, 70, 70, (x & 1) ^ (y & 1) ? 0xff80a0c0 : 0xffa0c0f0);
				piece = m_board.piece(x, y, &white);
				if(piece)
				{
					GRect r(white ? 50 : 0, 50 * (piece - 1), 50, 50);
					m_pImage->blitAlpha(70 * x + 10, 70 * y + 10, m_pPieces, &r);
				}
			}
		}
		if(m_lastMove[0] >= 0)
		{
			int yDest = m_lastMove[0];
			int xSrc = yDest % 8;
			yDest = (yDest - xSrc) / 8;
			int ySrc = yDest % 8;
			yDest = (yDest - ySrc) / 8;
			int xDest = yDest % 8;
			yDest = (yDest - xDest) / 8;
			m_pImage->arrow(70 * xSrc + 35, 70 * ySrc + 35, 70 * xDest + 35, 70 * yDest + 35, 0xff90b0d8, 10);
		}
		if(m_pieceInHand != GChessBoard::None)
		{
			m_pController->mousePos(&x, &y);
			x -= m_pCanvas->rect()->x;
			y -= m_pCanvas->rect()->y;
			GRect r(m_pieceInHandWhite ? 50 : 0, 50 * (m_pieceInHand - 1), 50, 50);
			m_pImage->blitAlpha(x - 25, y - 25, m_pPieces, &r);
		}
		int i;
		for(i = 0; i < m_hintMoveCount; i++)
		{
			x = 70 * m_hintMoves[2 * i] + 3;
			y = 70 * m_hintMoves[2 * i + 1] + 3;
			m_pImage->box(x, y, x + 63, y + 63, 0xffff0000);
		}
	}

	void PassTurn()
	{
		m_whitesTurn = !m_whitesTurn;
		if(++m_moves >= 1000)
		{
			m_winCounts[2]++;
			RestartGame();
		}
	}

	void DoMove(int xSrc, int ySrc, int xDest, int yDest)
	{
		if(m_board.move(xSrc, ySrc, xDest, yDest))
		{
			int score = EvaluateBoard(&m_board);
			if(score > 0)
				m_winCounts[0]++;
			else if(score < 0)
				m_winCounts[1]++;
			else
				m_winCounts[2]++;
				RestartGame();
		}
		SetLastMove(xSrc, ySrc, xDest, yDest);
		PassTurn();
	}

	void DoMouseDown(int x, int y)
	{
		if(m_pieceInHand != GChessBoard::None)
			DoMouseUp(x, y);
		x -= m_pCanvas->rect()->x;
		y -= m_pCanvas->rect()->y;
		x /= 70;
		y /= 70;
		if(x >= 0 && x < 8 && y >= 0 && y < 8)
		{
			m_pieceInHandX = x;
			m_pieceInHandY = y;
			m_pieceInHand = m_board.piece(x, y, &m_pieceInHandWhite);
			if(m_pieceInHand != GChessBoard::None)
			{
				m_hintMoveCount = m_board.moves(m_hintMoves, x, y);
				if(m_pieceInHandWhite != m_whitesTurn)
					m_pieceInHand = GChessBoard::None;
				else
					m_board.setPiece(x, y, GChessBoard::None, true);
			}
		}
	}

	void DoMouseUp(int x, int y)
	{
		x -= m_pCanvas->rect()->x;
		y -= m_pCanvas->rect()->y;
		x /= 70;
		y /= 70;
		if(m_pieceInHand != GChessBoard::None)
			m_board.setPiece(m_pieceInHandX, m_pieceInHandY, m_pieceInHand, m_pieceInHandWhite);
		if(x >= 0 && x < 8 && y >= 0 && y < 8 && m_pieceInHandWhite == m_whitesTurn && m_board.isValidMove(m_pieceInHandX, m_pieceInHandY, x, y) && !IsRepeatAggressiveMove(m_pieceInHandX, m_pieceInHandY, x, y))
			DoMove(m_pieceInHandX, m_pieceInHandY, x, y);
		m_pieceInHand = GChessBoard::None;
		m_hintMoveCount = 0;
	}

	void DoRandomMove()
	{
		int xSrc, ySrc, xDest, yDest;
		int xSrcPicked = 0;
		int ySrcPicked = 0;
		int xDestPicked = 0;
		int yDestPicked = 0;
		m_pMoveIterators[0].reset(&m_board, m_whitesTurn, m_prng);
		int moves = 0;
		while(m_pMoveIterators[0].nextMove(&xSrc, &ySrc, &xDest, &yDest))
		{
			moves++;
			if(m_prng->next(moves) == 0 && !IsRepeatAggressiveMove(xSrc, ySrc, xDest, yDest))
			{
				xSrcPicked = xSrc;
				ySrcPicked = ySrc;
				xDestPicked = xDest;
				yDestPicked = yDest;
			}
		}
		if(moves > 0)
			DoMove(xSrcPicked, ySrcPicked, xDestPicked, yDestPicked);
		else
			PassTurn();
	}

	int ComputeConservativePayoff(bool white, GChessBoard board, int xSrc, int ySrc, int xDest, int yDest, int cutoff, int depth)
	{
		board.move(xSrc, ySrc, xDest, yDest);
		if(depth >= m_maxDepth || depth >= MAX_DEPTH_EVER)
			return EvaluateBoard(&board);
		int score;
		int bestScore = (white ? -1000 : 1000);
		m_pMoveIterators[depth].reset(&board, white, m_prng);
		while(m_pMoveIterators[depth].nextMove(&xSrc, &ySrc, &xDest, &yDest))
		{
			score = ComputeConservativePayoff(!white, board, xSrc, ySrc, xDest, yDest, bestScore, depth + 1);
			if(white)
			{
				if(score > bestScore)
				{
					bestScore = score;
					if(bestScore >= cutoff)
						break;
				}
			}
			else
			{
				if(score < bestScore)
				{
					bestScore = score;
					if(bestScore <= cutoff)
						break;
				}
			}
		}
		return bestScore;
	}

	bool MaxiMinWithAlphaBetaPruning(bool white, int* pxSrc, int* pySrc, int* pxDest, int* pyDest)
	{
		m_boardEvals = 0;
		int xSrc, ySrc, xDest, yDest, score;
		int bestScore = (white ? -1000 : 1000);
		bool bGotOne = false;
		m_pMoveIterators[0].reset(&m_board, white, m_prng);
		while(m_pMoveIterators[0].nextMove(&xSrc, &ySrc, &xDest, &yDest))
		{
			if(IsRepeatAggressiveMove(xSrc, ySrc, xDest, yDest))
				continue;
			score = ComputeConservativePayoff(!white, m_board, xSrc, ySrc, xDest, yDest, bestScore, 1);
			if(white)
			{
				if(score > bestScore || !bGotOne)
				{
					bestScore = score;
					*pxSrc = xSrc; *pySrc = ySrc; *pxDest = xDest; *pyDest = yDest;
					bGotOne = true;
				}
			}
			else
			{
				if(score < bestScore || !bGotOne)
				{
					bestScore = score;
					*pxSrc = xSrc; *pySrc = ySrc; *pxDest = xDest; *pyDest = yDest;
					bGotOne = true;
				}
			}
		}
		if(white)
			m_lightEvals += m_boardEvals;
		else
			m_darkEvals += m_boardEvals;
		return bGotOne;
	}

	void DoAlphaBetaPruningMove(int depth)
	{
		m_maxDepth = depth;
		int xSrc = 0;
		int ySrc = 0;
		int xDest = 0;
		int yDest = 0;
		if(MaxiMinWithAlphaBetaPruning(m_whitesTurn, &xSrc, &ySrc, &xDest, &yDest))
			DoMove(xSrc, ySrc, xDest, yDest);
		else
			PassTurn();
	}

	void DoTurn()
	{
		int strategy = (m_whitesTurn ? m_pLightStrategy : m_pDarkStrategy)->selection();
		switch(strategy)
		{
			case 0: break;
			case 1: DoRandomMove(); break;
			case 2: DoAlphaBetaPruningMove(3); break;
			case 3: DoAlphaBetaPruningMove(6); break;
			default: GAssert(false); break;
		}
	}

	void Iterate()
	{
		DoTurn();
		DrawBoard();
		m_pCanvas->setImage(m_pImage);
	}

	virtual void onReleaseTextButton(GWidgetTextButton* pButton)
	{
		GAssert(false); // unknown button
	}

	int EvaluateBoard(GChessBoard* pBoard)
	{
		bool white;
		int score = 0;
		int x, y, value;
		GChessBoard::Piece piece;
		for(y = 0; y < 8; y++)
		{
			for(x = 0; x < 8; x++)
			{
				piece = pBoard->piece(x, y, &white);
				switch(piece)
				{
					case GChessBoard::None: value = 0; break;
					case GChessBoard::Pawn: value = 1; break;
					case GChessBoard::Rook: value = 5; break;
					case GChessBoard::Knight: value = 3; break;
					case GChessBoard::Bishop: value = 4; break;
					case GChessBoard::Queen: value = 9; break;
					case GChessBoard::King: value = 25; break;
					default: GAssert(false); value = 0; break;
				}
				if(white)
					score += value;
				else
					score -= value;
			}
		}
		m_boardEvals++;
		return score;
	}
};


// -------------------------------------------------------------------------------

class ChessView : public ViewBase
{
protected:
	ChessDialog* m_pDialog;

public:
	ChessView(ChessController* pController)
	: ViewBase()
	{
		m_pDialog = new ChessDialog(pController, m_screenRect.w, m_screenRect.h);
	}

	virtual ~ChessView()
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
		m_pDialog->DoMouseDown(x - m_screenRect.x, y - m_screenRect.y);
	}

	virtual void onMouseUp(int nButton, int x, int y)
	{
		m_pDialog->releaseButton(nButton);
		m_pDialog->DoMouseUp(x, y);
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


ChessController::ChessController()
: ControllerBase()
{
	m_pView = new ChessView(this);
}

ChessController::~ChessController()
{
	delete(m_pView);
}

void ChessController::RunModal()
{
	double timeOld = GTime::seconds();
	double time;
	m_pView->update();
	while(m_bKeepRunning)
	{
		time = GTime::seconds();
		handleEvents(time - timeOld);
		m_pView->update();
		GThread::sleep(150); // slow things down a bit
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
		ChessController c;
		c.RunModal();
	}
	catch(const std::exception& e)
	{
		cerr << e.what() << "\n";
		nRet = 1;
	}

	return nRet;
}

