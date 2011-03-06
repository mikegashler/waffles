/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#include "GError.h"
#include "GSparseMatrix.h"
#include <math.h>
#include "GMatrix.h"
#include "GVec.h"
#include "GFile.h"
#include "GRand.h"
#include <fstream>
#include "GTwt.h"

namespace GClasses {

GSparseMatrix::GSparseMatrix(size_t rows, size_t cols, double defaultValue)
: m_cols(cols), m_defaultValue(defaultValue)
{
	m_rows.resize(rows);
}

GSparseMatrix::GSparseMatrix(GTwtNode* pNode)
{
	m_defaultValue = pNode->field("def")->asDouble();
	m_cols = (size_t)pNode->field("cols")->asInt();
	GTwtNode* pRows = pNode->field("rows");
	size_t rows = pRows->itemCount();
	m_rows.resize(rows);
	for(size_t i = 0; i < rows; i++)
	{
		GTwtNode* pElements = pRows->item(i);
		for(size_t j = 0; j < pElements->itemCount(); j++)
		{
			size_t col = (size_t)pElements->item(j)->asInt();
			j++;
			double val = pElements->item(j)->asDouble();
			set(i, col, val);
		}
	}
}

GSparseMatrix::~GSparseMatrix()
{
}

GTwtNode* GSparseMatrix::toTwt(GTwtDoc* pDoc)
{
	GTwtNode* pNode = pDoc->newObj();
	pNode->addField(pDoc, "def", pDoc->newDouble(m_defaultValue));
	pNode->addField(pDoc, "cols", pDoc->newInt(m_cols));
	GTwtNode* pRows = pNode->addField(pDoc, "rows", pDoc->newList(m_rows.size()));
	for(size_t i = 0; i < m_rows.size(); i++)
	{
		GTwtNode* pElements = pRows->setItem(i, pDoc->newList(2 * rowNonDefValues(i)));
		size_t pos = 0;
		for(Iter it = rowBegin(i); it != rowEnd(i); it++)
		{
			pElements->setItem(pos++, pDoc->newInt(it->first));
			pElements->setItem(pos++, pDoc->newDouble(it->second));
		}
	}
	return pNode;
}

void GSparseMatrix::fullRow(double* pOutFullRow, size_t row)
{
	GVec::setAll(pOutFullRow, m_defaultValue, m_cols);
	Iter end = rowEnd(row);
	for(Iter it = rowBegin(row); it != end; it++)
		pOutFullRow[it->first] = it->second;
}

double GSparseMatrix::get(size_t row, size_t col)
{
	GAssert(row < m_rows.size() && col < m_cols); // out of range
	Map::iterator it = m_rows[row].find(col);
	if(it == m_rows[row].end())
		return m_defaultValue;
	return it->second;
}

void GSparseMatrix::set(size_t row, size_t col, double val)
{
	GAssert(row < m_rows.size() && col < m_cols); // out of range
	if(val == m_defaultValue)
		m_rows[row].erase(col);
	else
		m_rows[row][col] = val;
}

void GSparseMatrix::multiply(double scalar)
{
	if(m_defaultValue != 0.0)
		ThrowError("This method assumes the default value is 0");
	for(size_t r = 0; r < m_rows.size(); r++)
	{
		Map::iterator end = m_rows[r].end();
		for(Map::iterator it = m_rows[r].begin(); it != end; it++)
			it->second *= scalar;
	}
}

void GSparseMatrix::copyFrom(GSparseMatrix* that)
{
	size_t rows = std::min(m_rows.size(), that->rows());
	for(size_t r = 0; r < rows; r++)
	{
		Iter end = that->rowEnd(r);
		size_t pos = 0;
		for(Iter it = that->rowBegin(r); it != end && pos < m_cols; it++)
		{
			set(r, it->first, it->second);
			pos++;
		}
	}
}

void GSparseMatrix::copyFrom(GMatrix* that)
{
	size_t rows = std::min(m_rows.size(), that->rows());
	size_t cols = std::min(m_cols, (size_t)that->cols());
	for(size_t r = 0; r < rows; r++)
	{
		double* pRow = that->row(r);
		for(size_t c = 0; c < cols; c++)
		{
			set(r, c, *pRow);
			pRow++;
		}
	}
}

void GSparseMatrix::newRow()
{
	m_rows.resize(m_rows.size() + 1);
}

void GSparseMatrix::copyRow(Map& row)
{
	size_t n = m_rows.size();
	newRow();
	Map& m = m_rows[n];
	m = row;
}

GMatrix* GSparseMatrix::toFullMatrix()
{
	GMatrix* pData = new GMatrix(m_rows.size(), m_cols);
	for(size_t r = 0; r < m_rows.size(); r++)
	{
		double* pRow = pData->row(r);
		GVec::setAll(pRow, 0.0, m_cols);
		Iter end = rowEnd(r);
		for(Iter it = rowBegin(r); it != end; it++)
			pRow[it->first] = it->second;
	}
	return pData;
}

void GSparseMatrix::swapColumns(size_t a, size_t b)
{
	for(size_t r = 0; r < m_rows.size(); r++)
	{
		double aa = get(r, a);
		double bb = get(r, b);
		set(r, a, bb);
		set(r, b, aa);
	}
}

void GSparseMatrix::swapRows(size_t a, size_t b)
{
	std::swap(m_rows[a], m_rows[b]);
}

void GSparseMatrix::shuffle(GRand* pRand, GMatrix* pLabels)
{
	if(pLabels)
	{
		for(size_t n = m_rows.size(); n > 0; n--)
		{
			size_t r = (size_t)pRand->next(n);
			swapRows(r, n - 1);
			pLabels->swapRows(r, n - 1);
		}
	}
	else
	{
		for(size_t n = m_rows.size(); n > 0; n--)
			swapRows((size_t)pRand->next(n), n - 1);
	}
}

GSparseMatrix* GSparseMatrix::subMatrix(size_t row, size_t col, size_t height, size_t width)
{
	if(row + height >= m_rows.size() || col + width >= m_cols)
		ThrowError("out of range");
	GSparseMatrix* pSub = new GSparseMatrix(height, width, m_defaultValue);
	for(size_t y = 0; y < height; y++)
	{
		for(size_t x = 0; x < width; x++)
			pSub->set(y, x, get(row + y, col + x));
	}
	return pSub;
}
/*
double GSparseMatrix_pythag(double a, double b)
{
	double at = ABS(a);
	double bt = ABS(b);
	if(at > bt)
	{
		double ct = bt / at;
		return at * sqrt(1.0 + ct * ct);
	}
	else if(bt > 0.0)
	{
		double ct = at / bt;
		return bt * sqrt(1.0 + ct * ct);
	}
	else
		return 0.0;
}

double GSparseMatrix_takeSign(double a, double b)
{
	return (b >= 0.0 ? ABS(a) : -ABS(a));
}

void GSparseMatrix::singularValueDecomposition(GSparseMatrix** ppU, double** ppDiag, GSparseMatrix** ppV, int maxIters)
{
	if(rows() >= cols())
		singularValueDecompositionHelper(ppU, ppDiag, ppV, maxIters);
	else
	{
		transpose();
		singularValueDecompositionHelper(ppV, ppDiag, ppU, maxIters);
		transpose();
		(*ppV)->transpose();
		(*ppU)->transpose();
	}
}

void GSparseMatrix::singularValueDecompositionHelper(GSparseMatrix** ppU, double** ppDiag, GSparseMatrix** ppV, int maxIters)
{
	if(m_defaultValue != 0.0)
		ThrowError("This method assumes the default value is 0");
	int m = rows();
	int n = cols();
	if(m < n)
		ThrowError("Expected at least as many rows as columns");
	int i, j, k;
	int l = 0;
	int q, iter;
	double c, f, h, s, x, y, z;
	double norm = 0.0;
	double g = 0.0;
	double scale = 0.0;
	GSparseMatrix* pU = new GSparseMatrix(m, m, 0.0);
	Holder<GSparseMatrix> hU(pU);
	pU->copyFrom(this);
	double* pSigma = new double[n];
	ArrayHolder<double> hSigma(pSigma);
	GSparseMatrix* pV = new GSparseMatrix(n, n, 0.0);
	Holder<GSparseMatrix> hV(pV);
	GTEMPBUF(double, temp, n + m);
	double* temp2 = temp + n;

	// Householder reduction to bidiagonal form
	for(int i = 0; i < n; i++)
	{
		// Left-hand reduction
		temp[i] = scale * g;
		l = i + 1;
		g = 0.0;
		s = 0.0;
		scale = 0.0;
		if(i < m)
		{
			Iter kend = pU->colEnd(i);
			for(Iter kk = pU->colBegin(i); kk != kend; kk++)
			{
				if(kk->first >= (unsigned int)i)
					scale += ABS(kk->second);
			}
			if(scale != 0.0)
			{
				for(Iter kk = pU->colBegin(i); kk != kend; kk++)
				{
					if(kk->first >= (unsigned int)i)
					{
						double t = kk->second / scale;
						pU->set(kk->first, i, t);
						s += (t * t);
					}
				}
				f = pU->get(i, i);
				g = -GSparseMatrix_takeSign(sqrt(s), f);
				h = f * g - s;
				pU->set(i, i, f - g);
				if(i != n - 1)
				{
					for(j = l; j < n; j++)
					{
						s = 0.0;
						for(Iter kk = pU->colBegin(i); kk != kend; kk++)
						{
							if(kk->first >= (unsigned int)i)
								s += kk->second * pU->get(kk->first, j);
						}
						f = s / h;
						for(Iter kk = pU->colBegin(i); kk != kend; kk++)
						{
							if(kk->first >= (unsigned int)i)
								pU->set(kk->first, j, pU->get(kk->first, j) + f * kk->second);
						}
					}
				}
				for(Iter kk = pU->colBegin(i); kk != kend; kk++)
				{
					if(kk->first >= (unsigned int)i)
						pU->set(kk->first, i, pU->get(kk->first, i) * scale);
				}
			}
		}
		pSigma[i] = scale * g;

		// Right-hand reduction
		g = 0.0;
		s = 0.0;
		scale = 0.0;
		if(i < m && i != n - 1) 
		{
			Iter kend = pU->rowEnd(i);
			for(Iter kk = pU->rowBegin(i); kk != kend; kk++)
			{
				if(kk->first >= (unsigned int)n)
					break;
				if(kk->first >= (unsigned int)l)
					scale += ABS(kk->second);
			}
			if(scale != 0.0) 
			{
				for(Iter kk = pU->rowBegin(i); kk != kend; kk++)
				{
					if(kk->first >= (unsigned int)n)
						break;
					if(kk->first >= (unsigned int)l)
					{
						double t = kk->second / scale;
						pU->set(i, kk->first, t);
						s += (t * t);
					}
				}
				f = pU->get(i, l);
				g = -GSparseMatrix_takeSign(sqrt(s), f);
				h = f * g - s;
				pU->set(i, l, f - g);
				for(k = l; k < n; k++)
					temp[k] = pU->get(i, k) / h;
				if(i != m - 1) 
				{
					for(j = l; j < m; j++) 
					{
						s = 0.0;
						for(Iter kk = pU->rowBegin(i); kk != kend; kk++)
						{
							if(kk->first >= (unsigned int)n)
								break;
							if(kk->first >= (unsigned int)l)
								s += pU->get(j, kk->first) * kk->second;
						}
						Iter kend2 = pU->rowEnd(j);
						for(Iter kk = pU->rowBegin(j); kk != kend2; kk++)
						{
							if(kk->first >= (unsigned int)n)
								break;
							if(kk->first >= (unsigned int)l)
								pU->set(j, kk->first, pU->get(j, kk->first) + s * temp[kk->first]);
						}
					}
				}
				for(Iter kk = pU->rowBegin(i); kk != kend; kk++)
				{
					if(kk->first >= (unsigned int)n)
						break;
					if(kk->first >= (unsigned int)l)
						pU->set(i, kk->first, kk->second * scale);
				}
			}
		}
		norm = MAX(norm, ABS(pSigma[i]) + ABS(temp[i]));
	}

	// Accumulate right-hand transform
	for(int i = n - 1; i >= 0; i--)
	{
		if(i < n - 1)
		{
			if(g != 0.0)
			{
				Iter jend = pU->rowEnd(i);
				for(Iter jj = pU->rowBegin(i); jj != jend; jj++)
				{
					if(jj->first >= (unsigned int)n)
						break;
					if(jj->first >= (unsigned int)l)
						pV->set(i, jj->first, (jj->second / pU->get(i, l)) / g); // (double-division to avoid underflow)
				}
				for(j = l; j < n; j++)
				{
					s = 0.0;
					Iter kend = pU->rowEnd(i);
					for(Iter kk = pU->rowBegin(i); kk != kend; kk++)
					{
						if(kk->first >= (unsigned int)n)
							break;
						if(kk->first >= (unsigned int)l)
							s += kk->second * pV->get(j, kk->first);
					}
					kend = pV->rowEnd(i);
					for(Iter kk = pV->rowBegin(i); kk != kend; kk++)
					{
						if(kk->first >= (unsigned int)n)
							break;
						if(kk->first >= (unsigned int)l)
							pV->set(j, kk->first, pV->get(j, kk->first) + s * kk->second);
					}
				}
			}
			for(j = l; j < n; j++)
			{
				pV->set(i, j, 0.0);
				pV->set(j, i, 0.0);
			}
		}
		pV->set(i, i, 1.0);
		g = temp[i];
		l = i;
	}

	// Accumulate left-hand transform
	for(i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = pSigma[i];
		if(i < n - 1)
		{
			for(j = l; j < n; j++)
				pU->set(i, j, 0.0);
		}
		if(g != 0.0)
		{
			g = 1.0 / g;
			if(i != n - 1)
			{
				for(j = l; j < n; j++)
				{
					s = 0.0;
					Iter kend = pU->colEnd(i);
					for(Iter kk = pU->colBegin(i); kk != kend; kk++)
					{
						if(kk->first >= (unsigned int)l)
							s += kk->second * pU->get(kk->first, j);
					}
					f = (s / pU->get(i, i)) * g;
					if(f != 0.0)
					{
						for(Iter kk = pU->colBegin(i); kk != kend; kk++)
						{
							if(kk->first >= (unsigned int)i)
								pU->set(kk->first, j, pU->get(kk->first, j) + f * kk->second);
						}
					}
				}
			}
			for(j = i; j < m; j++)
				pU->set(j, i, pU->get(j, i) * g);
		} 
		else 
		{
			for(j = i; j < m; j++)
				pU->set(j, i, 0.0);
		}
		pU->set(i, i, pU->get(i, i) + 1.0);
	}

	// Diagonalize the bidiagonal matrix
	for(k = n - 1; k >= 0; k--) // For each singular value
	{
		for(iter = 1; iter <= maxIters; iter++)
		{
			// Test for splitting
			bool flag = true;
			for(l = k; l >= 0; l--)
			{
				q = l - 1;
				if(ABS(temp[l]) + norm == norm)
				{
					flag = false;
					break;
				}
				if(ABS(pSigma[q]) + norm == norm)
					break;
			}

			if(flag)
			{
				c = 0.0;
				s = 1.0;
				for(i = l; i <= k; i++)
				{
					f = s * temp[i];
					temp[i] *= c;
					if(ABS(f) + norm == norm)
						break;
					g = pSigma[i];
					h = GSparseMatrix_pythag(f, g);
					pSigma[i] = h;
					h = 1.0 / h;
					c = g * h;
					s = -f * h;
					Iter jendi = pU->colEnd(i);
					Iter jendq = pU->colEnd(q);
					Iter jji = pU->colBegin(i);
					Iter jjq = pU->colBegin(q);
					int tpos;
					for(tpos = 0; jji != jendi || jjq != jendq; tpos++)
					{
						if(jjq == jendq || (jji != jendi && jji->first < jjq->first))
						{
							temp2[tpos] = jji->first;
							jji++;
						}
						else
						{
							temp2[tpos] = jjq->first;
							if(jji != jendi && jjq->first == jji->first)
								jji++;
							jjq++;
						}
					}
					for(int tpos2 = 0; tpos2 < tpos; tpos2++)
					{
						y = pU->get((unsigned int)temp2[tpos2], q);
						z = pU->get((unsigned int)temp2[tpos2], i);
						pU->set((unsigned int)temp2[tpos2], q, y * c + z * s);
						pU->set((unsigned int)temp2[tpos2], i, z * c - y * s);
					}
				}
			}

			z = pSigma[k];
			if(l == k)
			{
				// Detect convergence
				if(z < 0.0)
				{
					// Singular value should be positive
					pSigma[k] = -z;
					for(j = 0; j < n; j++)
						pV->set(k, j, pV->get(k, j) * -1.0);
				}
				break;
			}
			if(iter >= maxIters)
				ThrowError("failed to converge");

			// Shift from bottom 2x2 minor
			x = pSigma[l];
			q = k - 1;
			y = pSigma[q];
			g = temp[q];
			h = temp[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = GSparseMatrix_pythag(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + GSparseMatrix_takeSign(g, f))) - h)) / x;

			// QR transform
			c = 1.0;
			s = 1.0;
			for(j = l; j <= q; j++)
			{
				i = j + 1;
				g = temp[i];
				y = pSigma[i];
				h = s * g;
				g = c * g;
				z = GSparseMatrix_pythag(f, h);
				temp[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				Iter pendi = pV->rowEnd(i);
				Iter pendj = pV->rowEnd(j);
				Iter ppi = pV->rowBegin(i);
				Iter ppj = pV->rowBegin(j);
				int tpos;
				for(tpos = 0; ppi != pendi || ppj != pendj; tpos++)
				{
					if(ppj == pendj || (ppi != pendi && ppi->first < ppj->first))
					{
						temp2[tpos] = ppi->first;
						ppi++;
					}
					else
					{
						temp2[tpos] = ppj->first;
						if(ppi != pendi && ppj->first == ppi->first)
							ppi++;
						ppj++;
					}
				}
				for(int tpos2 = 0; tpos2 < tpos; tpos2++)
				{
					x = pV->get(j, (unsigned int)temp2[tpos2]);
					z = pV->get(i, (unsigned int)temp2[tpos2]);
					pV->set(j, (unsigned int)temp2[tpos2], x * c + z * s);
					pV->set(i, (unsigned int)temp2[tpos2], z * c - x * s);
				}
				z = GSparseMatrix_pythag(f, h);
				pSigma[j] = z;
				if(z != 0.0)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = c * g + s * y;
				x = c * y - s * g;
				pendi = pU->colEnd(i);
				pendj = pU->colEnd(j);
				ppi = pU->colBegin(i);
				ppj = pU->colBegin(j);
				for(tpos = 0; ppi != pendi || ppj != pendj; tpos++)
				{
					if(ppj == pendj || (ppi != pendi && ppi->first < ppj->first))
					{
						temp2[tpos] = ppi->first;
						ppi++;
					}
					else
					{
						temp2[tpos] = ppj->first;
						if(ppi != pendi && ppj->first == ppi->first)
							ppi++;
						ppj++;
					}
				}
				for(int tpos2 = 0; tpos2 < tpos; tpos2++)
				{
					y = pU->get((unsigned int)temp2[tpos2], j);
					z = pU->get((unsigned int)temp2[tpos2], i);
					pU->set((unsigned int)temp2[tpos2], j, y * c + z * s);
					pU->set((unsigned int)temp2[tpos2], i, z * c - y * s);
				}
			}
			temp[l] = 0.0;
			temp[k] = f;
			pSigma[k] = x;
		}
	}

	// Sort the singular values from largest to smallest
	for(i = 1; i < n; i++)
	{
		for(j = i; j > 0; j--)
		{
			if(pSigma[j - 1] >= pSigma[j])
				break;
			pU->swapColumns(j - 1, j);
			pV->swapRows(j - 1, j);
			std::swap(pSigma[j - 1], pSigma[j]);
		}
	}

	// Return results
	*ppU = hU.release();
	*ppDiag = hSigma.release();
	*ppV = hV.release();
}
*/
#ifndef NO_TEST_CODE
// static
void GSparseMatrix::test()
{
/*
	// Make the data
	GSparseMatrix sm(4, 4);
	sm.set(0, 0, 2.0); sm.set(0, 2, 3.0);
	sm.set(1, 0, 1.0); sm.set(2, 3, -2.0);
	sm.set(2, 2, 5.0);
	sm.set(3, 1, -3.0); sm.set(3, 3, -1.0);
	GMatrix* fm = sm.toFullMatrix();
	Holder<GMatrix> hFM(fm);

	// Do it with the full matrix
	GMatrix* pU;
	double* pDiag;
	GMatrix* pV;
	fm->singularValueDecomposition(&pU, &pDiag, &pV);
	Holder<GMatrix> hU(pU);
	ArrayHolder<double> hDiag(pDiag);
	Holder<GMatrix> hV(pV);

	// Do it with the sparse matrix
	GSparseMatrix* pSU;
	double* pSDiag;
	GSparseMatrix* pSV;
	sm.singularValueDecomposition(&pSU, &pSDiag, &pSV);
	Holder<GSparseMatrix> hSU(pSU);
	ArrayHolder<double> hSDiag(pSDiag);
	Holder<GSparseMatrix> hSV(pSV);

	// Check the results
	GMatrix* pV2 = pSV->toFullMatrix();
	Holder<GMatrix> hV2(pV2);
	double err = pV2->sumSquaredDifference(*pV, false);
	if(err > 1e-6)
		ThrowError("Failed");
*/
}
#endif

} // namespace GClasses

