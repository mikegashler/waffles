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

#ifndef __GCHESS_H__
#define __GCHESS_H__

#include <stdio.h>
#include <string.h>
#include <iostream>

namespace GClasses {

#define MAX_PIECE_MOVES 27

class GRand;

/// Represents the state of a chess board, and provides
/// some basic functionality for implementing a chess game.
class GChessBoard
{
public:
	enum Piece
	{
		None = 0,
		Pawn = 1,
		Rook = 2,
		Knight = 3,
		Bishop = 4,
		Queen = 5,
		King = 6,
		PieceMask = 7,
		WhiteMask = 8,
		AllMask = 15,
	};

protected:
	unsigned int m_rows[8];

public:
	GChessBoard()
	{
		resetBoard();
	}

	/// copy-constructor
	GChessBoard(const GChessBoard& that)
	{
		memcpy(m_rows, that.m_rows, sizeof(int) * 8);
	}

	~GChessBoard()
	{
	}

	/// Returns the piece located at position (col, row), and sets *pbWhite to true
	/// if the piece is white.
	inline Piece piece(int col, int row, bool* pbWhite)
	{
		unsigned int i = (m_rows[row] >> (4 * col));
		*pbWhite = ((i & WhiteMask) ? true : false);
		return (Piece)(i & PieceMask);
	}

	/// Sets the piece at location (col, row). If p is None, then it doesn't
	/// matter what the value of white is.
	inline void setPiece(int col, int row, Piece p, bool white)
	{
		m_rows[row] &= (~(AllMask << (4 * col)));
		m_rows[row] |= ((p | (white ? WhiteMask : 0)) << (4 * col));
	}

	/// Sets up the board for a new game
	void resetBoard();

	/// Evaluates the board without looking any deeper
	int heuristic();

	/// Returns true iff the parameters represent a valid move
	bool isValidMove(int xSrc, int ySrc, int xDest, int yDest);

	/// Print a representation of the board to the specified stream
	void printBoard(std::ostream& stream);

	/// Pass in the coordinates of a square with a piece on it
	/// and it will return the number of moves that piece can make
	/// and put the coordinates of the moves in pOutMoves (first
	/// col, then row, so pOutMoves should be big enough to hold
	/// 2 * MAX_PIECE_MOVES values)
	int moves(int* pOutMoves, int col, int row);

	/// Moves the piece from (xSrc, ySrc) to (xDest, yDest). If this move
	/// takes a king, then it will remove all pieces of the same color as
	/// the king that was taken and return true to indicate that the move
	/// ended the game.
	bool move(int xSrc, int ySrc, int xDest, int yDest);

protected:
	inline int inc(int pos)
	{
		if(pos < 0 || pos >= 7)
			return -1;
		return pos + 1;
	}

	inline int dec(int pos)
	{
		if(pos < 1)
			return -1;
		return pos -1;
	}

	bool checkMove(int* pOutMoves, int* pnMoves, int col, int row, bool bWhite);
	bool checkPawnMove(int* pOutMoves, int* pnMoves, int col, int row, bool bDiagonal, bool bWhite);
};


/// Iterates through all the possible moves for the specified color.
class GChessMoveIterator
{
protected:
	int m_n, m_move, m_moveCount;
	int m_moves[2 * MAX_PIECE_MOVES];
	GChessBoard* m_pBoard;
	bool m_white;

public:
	/// Constructs a move iterator. (Doesn't call Reset, so you must still call Reset before you call GetNextMove).
	GChessMoveIterator();
	~GChessMoveIterator();

	/// Resets the iterator and specifies the color whose moves you wish to iterate.
	void reset(GChessBoard* pBoard, bool white);

	/// Gets the next possible move for the specified color. Returns false if there are no more moves (so the
	/// values in xSrc, ySrc, xDest, and yDest are bogus)
	bool nextMove(int* xSrc, int* ySrc, int* xDest, int* yDest);
};

} // namespace GClasses

#endif // __GCHESS_H__
