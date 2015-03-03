/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
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

#include "GChess.h"
#include <GClasses/GError.h>

using namespace GClasses;
using std::ostream;

void GChessBoard::resetBoard()
{
	int i, j;
	setPiece(0, 0, Rook, true);
	setPiece(1, 0, Knight, true);
	setPiece(2, 0, Bishop, true);
	setPiece(3, 0, Queen, true);
	setPiece(4, 0, King, true);
	setPiece(5, 0, Bishop, true);
	setPiece(6, 0, Knight, true);
	setPiece(7, 0, Rook, true);
	for(i = 0; i < 8; i++)
		setPiece(i, 1, Pawn, true);
	for(j = 2; j < 6; j++)
	{
		for(i = 0; i < 8; i++)
			setPiece(i, j, None, false);
	}
	for(i = 0; i < 8; i++)
		setPiece(i, 6, Pawn, false);
	setPiece(0, 7, Rook, false);
	setPiece(1, 7, Knight, false);
	setPiece(2, 7, Bishop, false);
	setPiece(3, 7, Queen, false);
	setPiece(4, 7, King, false);
	setPiece(5, 7, Bishop, false);
	setPiece(6, 7, Knight, false);
	setPiece(7, 7, Rook, false);
}

int GChessBoard::heuristic()
{
	bool white;
	int score = 0;
	int x, y, value;
	GChessBoard::Piece p;
	for(y = 0; y < 8; y++)
	{
		for(x = 0; x < 8; x++)
		{
			p = piece(x, y, &white);
			switch(p)
			{
				case GChessBoard::None: value = 0; break;
				case GChessBoard::Pawn: value = 10; break;
				case GChessBoard::Rook: value = 63; break;
				case GChessBoard::Knight: value = 31; break;
				case GChessBoard::Bishop: value = 36; break;
				case GChessBoard::Queen: value = 88; break;
				case GChessBoard::King: value = 500; break;
				default: GAssert(false); value = 0; break;
			}
			if(white)
				score += value;
			else
				score -= value;
		}
	}
	return score;
}

bool GChessBoard::checkMove(int* pOutMoves, int* pnMoves, int col, int row, bool bWhite)
{
	if(col < 0 || row < 0)
		return true;
	bool white;
	Piece p = piece(col, row, &white);
	if(p > 0 && white == bWhite)
		return true;
	int index = (*pnMoves) * 2;
	pOutMoves[index] = col;
	pOutMoves[index + 1] = row;
	(*pnMoves)++;
	return (p > 0);
}

bool GChessBoard::checkPawnMove(int* pOutMoves, int* pnMoves, int col, int row, bool bDiagonal, bool bWhite)
{
	if(col < 0 || row < 0)
		return true;
	bool white;
	Piece p = piece(col, row, &white);
	if(bDiagonal)
	{
		if(p == None || white == bWhite)
			return true;
	}
	else
	{
		if(p > 0)
			return true;
	}
	int index = (*pnMoves) * 2;
	pOutMoves[index] = col;
	pOutMoves[index + 1] = row;
	(*pnMoves)++;
	return (p > 0);
}

int GChessBoard::moves(int* pOutMoves, int col, int row)
{
	bool bWhite;
	Piece p = piece(col, row, &bWhite);
	int nMoves = 0;
	int i, j;
	switch(p)
	{
		case Pawn:
			if(bWhite)
			{
				if(!checkPawnMove(pOutMoves, &nMoves, col, inc(row), false, bWhite) && row == 1)
					checkPawnMove(pOutMoves, &nMoves, col, inc(inc(row)), false, bWhite);
				checkPawnMove(pOutMoves, &nMoves, inc(col), inc(row), true, bWhite);
				checkPawnMove(pOutMoves, &nMoves, dec(col), inc(row), true, bWhite);
			}
			else
			{
				if(!checkPawnMove(pOutMoves, &nMoves, col, dec(row), false, bWhite) && row == 6)
					checkPawnMove(pOutMoves, &nMoves, col, dec(dec(row)), false, bWhite);
				checkPawnMove(pOutMoves, &nMoves, inc(col), dec(row), true, bWhite);
				checkPawnMove(pOutMoves, &nMoves, dec(col), dec(row), true, bWhite);
			}
			break;
		case Bishop:
			for(i = inc(col), j=inc(row); true; i = inc(i), j = inc(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = dec(col), j=inc(row); true; i = dec(i), j = inc(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = inc(col), j=dec(row); true; i = inc(i), j = dec(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = dec(col), j=dec(row); true; i = dec(i), j = dec(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			break;
		case Knight:
			checkMove(pOutMoves, &nMoves, inc(inc(col)), inc(row), bWhite);
			checkMove(pOutMoves, &nMoves, inc(col), inc(inc(row)), bWhite);
			checkMove(pOutMoves, &nMoves, dec(col), inc(inc(row)), bWhite);
			checkMove(pOutMoves, &nMoves, dec(dec(col)), inc(row), bWhite);
			checkMove(pOutMoves, &nMoves, dec(dec(col)), dec(row), bWhite);
			checkMove(pOutMoves, &nMoves, dec(col), dec(dec(row)), bWhite);
			checkMove(pOutMoves, &nMoves, inc(col), dec(dec(row)), bWhite);
			checkMove(pOutMoves, &nMoves, inc(inc(col)), dec(row), bWhite);
			break;
		case Rook:
			for(i = inc(col); true; i = inc(i))
				if(checkMove(pOutMoves, &nMoves, i, row, bWhite))
					break;
			for(i = dec(col); true; i = dec(i))
				if(checkMove(pOutMoves, &nMoves, i, row, bWhite))
					break;
			for(j = inc(row); true; j = inc(j))
				if(checkMove(pOutMoves, &nMoves, col, j, bWhite))
					break;
			for(j = dec(row); true; j = dec(j))
				if(checkMove(pOutMoves, &nMoves, col, j, bWhite))
					break;
			break;
		case Queen:
			for(i = inc(col); true; i = inc(i))
				if(checkMove(pOutMoves, &nMoves, i, row, bWhite))
					break;
			for(i = dec(col); true; i = dec(i))
				if(checkMove(pOutMoves, &nMoves, i, row, bWhite))
					break;
			for(j = inc(row); true; j = inc(j))
				if(checkMove(pOutMoves, &nMoves, col, j, bWhite))
					break;
			for(j = dec(row); true; j = dec(j))
				if(checkMove(pOutMoves, &nMoves, col, j, bWhite))
					break;
			for(i = inc(col), j=inc(row); true; i = inc(i), j = inc(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = dec(col), j=inc(row); true; i = dec(i), j = inc(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = inc(col), j=dec(row); true; i = inc(i), j = dec(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			for(i = dec(col), j=dec(row); true; i = dec(i), j = dec(j))
				if(checkMove(pOutMoves, &nMoves, i, j, bWhite))
					break;
			break;
		case King:
			checkMove(pOutMoves, &nMoves, inc(col), row, bWhite);
			checkMove(pOutMoves, &nMoves, inc(col), inc(row), bWhite);
			checkMove(pOutMoves, &nMoves, col, inc(row), bWhite);
			checkMove(pOutMoves, &nMoves, dec(col), inc(row), bWhite);
			checkMove(pOutMoves, &nMoves, dec(col), row, bWhite);
			checkMove(pOutMoves, &nMoves, dec(col), dec(row), bWhite);
			checkMove(pOutMoves, &nMoves, col, dec(row), bWhite);
			checkMove(pOutMoves, &nMoves, inc(col), dec(row), bWhite);
			break;
		default:
			break;
	}
	return nMoves;
}

bool GChessBoard::isValidMove(int xSrc, int ySrc, int xDest, int yDest)
{
	int mvs[2 * MAX_PIECE_MOVES];
	int moveCount = moves(mvs, xSrc, ySrc);
	int i;
	for(i = 0; i < moveCount; i++)
	{
		if(mvs[2 * i] == xDest && mvs[2 * i + 1] == yDest)
			return true;
	}
	return false;
}

void GChessBoard::printBoard(ostream& stream)
{
	int i, j;
	bool white;
	stream << "  A  B  C  D  E  F  G  H\n";
	stream << " +";
	for(j = 0; j < 8; j++)
		stream << "--+";
	for(j = 7; j >= 0; j--)
	{
		stream << ('1' + j) << "|";
		for(i = 0; i < 8; i++)
		{
			Piece p = piece(i, j, &white);
			if(p != None)
			{
				if(white)
					stream << "w";
				else
					stream << "b";
			}
			switch(p)
			{
				case None: stream << "[]"; break;
				case Pawn: stream << "p"; break;
				case Rook: stream << "r"; break;
				case Knight: stream << "n"; break;
				case Bishop: stream << "b"; break;
				case Queen: stream << "q"; break;
				case King: stream << "K"; break;
				default: stream << "?"; break;
			}
			stream << "|";
		}
		stream << ('1' + j) << "\n +";
		for(j = 0; j < 8; j++)
			stream << "--+";
		stream << "\n";
	}
	stream << "  A  B  C  D  E  F  G  H\n";
}

bool GChessBoard::move(int xSrc, int ySrc, int xDest, int yDest)
{
	GAssert(xSrc >= 0 && xSrc < 8 && ySrc >= 0 && ySrc < 8);
	GAssert(xDest >= 0 && xDest < 8 && yDest >= 0 && yDest < 8);
	bool white, targetWhite;
	GChessBoard::Piece target = piece(xDest, yDest, &targetWhite);
	GChessBoard::Piece p = piece(xSrc, ySrc, &white);
	GAssert(p != None);
	GAssert(target == None || white != targetWhite); // can't take your own piece
	if(p == GChessBoard::Pawn && (yDest == 0 || yDest == 7))
		p = GChessBoard::Queen; // a pawn that crosses the board becomes a queen
	setPiece(xDest, yDest, p, white);
	setPiece(xSrc, ySrc, GChessBoard::None, true);
	if(target == King)
	{
		// If you take the opponent's king, remove all of the opponent's pieces. This
		// makes sure that look-ahead strategies don't try to look beyond the end of
		// the game (example: sacrifice a king for a king and some other piece.)
		int x, y;
		for(y = 0; y < 8; y++)
		{
			for(x = 0; x < 8; x++)
			{
				if(piece(x, y, &white) != None && white == targetWhite)
					setPiece(x, y, None, true);
			}
		}
		return true;
	}
	return false;
}

// --------------------------------------------------------------------

GChessMoveIterator::GChessMoveIterator()
{
	m_pBoard = NULL;
	m_white = true;
}

GChessMoveIterator::~GChessMoveIterator()
{
}

void GChessMoveIterator::reset(GChessBoard* pBoard, bool white)
{
	m_pBoard = pBoard;
	m_white = white;
	m_n = -1;
	m_move = 0;
	m_moveCount = 0;
}

bool GChessMoveIterator::nextMove(int* xSrc, int* ySrc, int* xDest, int* yDest)
{
	if(m_n >= 64)
		return false;
	bool white;
	int x, y;
	while(true)
	{
		if(m_move < m_moveCount)
		{
			*xSrc = m_n % 8;
			*ySrc = m_n / 8;
			*xDest = m_moves[2 * m_move];
			*yDest = m_moves[2 * m_move + 1];
			GAssert(*xSrc >= 0 && *xSrc < 8 && *ySrc >= 0 && *ySrc < 8); // out of range
			GAssert(*xDest >= 0 && *xDest < 8 && *yDest >= 0 && *yDest < 8); // out of range
			m_move++;
			return true;
		}
		else
		{
			if(++m_n >= 64)
				return false;
			x = m_n % 8;
			y = m_n / 8;
			if(m_pBoard->piece(x, y, &white) != GChessBoard::None && white == m_white)
			{
				m_move = 0;
				m_moveCount = m_pBoard->moves(m_moves, x, y);
			}
		}
	}
}
