/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
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

	/// Sets the piece at location (col, row). If piece is None, then it doesn't
	/// matter what the value of white is.
	inline void setPiece(int col, int row, Piece piece, bool white)
	{
		m_rows[row] &= (~(AllMask << (4 * col)));
		m_rows[row] |= ((piece | (white ? WhiteMask : 0)) << (4 * col));
	}

	/// Sets up the board for a new game
	void resetBoard();

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


/// Iterates through all the possible moves for the specified color. It iterates through the pieces in
/// a random order. It also iterates through the moves for each piece in a random order, but it will
/// visit each move for the current piece before considering the next piece.
class GChessMoveIterator
{
protected:
	int m_n, m_move, m_moveCount;
	int m_moves[2 * MAX_PIECE_MOVES];
	int m_order[64];
	GChessBoard* m_pBoard;
	bool m_white;
	GRand* m_pRand;

public:
	/// Constructs a move iterator. (Doesn't call Reset, so you must still call Reset before you call GetNextMove).
	GChessMoveIterator();
	~GChessMoveIterator();

	/// Resets the iterator and specifies the color whose moves you wish to iterate.
	void reset(GChessBoard* pBoard, bool white, GRand* pRand);

	/// Gets the next possible move for the specified color. Returns false if there are no more moves (so the
	/// values in xSrc, ySrc, xDest, and yDest are bogus)
	bool nextMove(int* xSrc, int* ySrc, int* xDest, int* yDest);
};

} // namespace GClasses

#endif // __GCHESS_H__
