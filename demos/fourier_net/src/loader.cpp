#include <GClasses/GRand.h>
#include <math.h>
#include "loader.h"

#define TRAIN_SIZE	100
#define TEST_SIZE	100

void Loader::loadStockData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	// 3   - AAPL
	// 108 - MSFT
	// 157 - DJI
	
	GMatrix raw;
	raw.loadArff("data/stocks.arff");
	
	size_t dims = 1;
	size_t offset = 600;
	size_t train_size = 500;
	size_t test_size = 300;
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	double log_10 = log(10);
	double vert_offset = 3.9;
	double scale = 20;
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		x[0] = double(i) / train_size;
		y[0] = ((log(raw[offset + i][157]) / log_10) - vert_offset) * scale;
	}
}

void Loader::loadPrecipitationData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	// Only 87 samples, so TEST_SIZE should be 23 if TRAIN_SIZE is 64
	
	GMatrix raw;
	raw.loadArff("data/precipitation.arff");
	
	size_t dims = 1;
	double *x, *y;
	
	trainFeat.resize(TRAIN_SIZE, 1);
	trainLab.resize(TRAIN_SIZE, dims);
	testFeat.resize(TEST_SIZE, 1);
	testLab.resize(TEST_SIZE, dims);
	
	for(size_t i = 0; i < TRAIN_SIZE + TEST_SIZE; i++)
	{
		if(i < TRAIN_SIZE)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - TRAIN_SIZE];
			y = testLab[i - TRAIN_SIZE];
		}
		
		x[0] = double(i) / TRAIN_SIZE;
		y[0] = raw[i][1] * 0.05;
		// y[1] = raw[i][2] * 0.05;
		// y[2] = raw[i][3] * 0.05;
		// y[3] = raw[i][4] * 0.05;
	}
	
	for(size_t i = 1; i < TRAIN_SIZE - 1; i++)
	{
		for(size_t j = 0; j < dims; j++)
		{
			trainLab[i][j] = (trainLab[i-1][j] + 2 * trainLab[i][j] + trainLab[i+1][j]) * 0.25;
		}
	}
}

void Loader::loadTemperatureData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	// TODO: determine optimal granularity
	
	GMatrix raw;
	raw.loadArff("data/melbourne-temperature.arff");
	
	size_t dims = 2;
	size_t offset = 0;
	size_t granularity = 7;
	
	double *x, *y;
	
	trainFeat.resize(TRAIN_SIZE, 1);
	trainLab.resize(TRAIN_SIZE, dims);
	testFeat.resize(TEST_SIZE, 1);
	testLab.resize(TEST_SIZE, dims);
	
	for(size_t i = 0; i < TRAIN_SIZE + TEST_SIZE; i++)
	{
		if(i < TRAIN_SIZE)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - TRAIN_SIZE];
			y = testLab[i - TRAIN_SIZE];
		}
		
		x[0] = double(i) / TRAIN_SIZE;
		
		for(size_t j = 0; j < dims; j++)
		{
			y[j] = 0.0;
			for(size_t k = granularity; k < granularity * 2; k++)
			{
				y[j] += raw[offset + i * granularity + k][j + 1];
			}
			y[j] /= double(granularity);
			y[j] *= 0.05;
		}
	}
	
	// Smooth input data
	
	for(size_t i = 1; i < TRAIN_SIZE - 1; i++)
	{
		for(size_t j = 0; j < dims; j++)
		{
			trainLab[i][j] = (trainLab[i-1][j] + trainLab[i][j] + trainLab[i+1][j]) / 3.0;
		}
	}
}

void Loader::loadLaborData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/labor_stats.arff");
	
	size_t dims = 1;
	size_t offset = 0;
	size_t train_size = 258;
	size_t test_size = 96;
	
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		*y = raw[offset + i][0];
	}
}

void Loader::loadLaborData2(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/labor_stats.arff");
	
	size_t dims = 1;
	size_t offset = 0;
	size_t train_size = 258;
	size_t test_size = 96;
	
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		*y = raw[offset + i][0];
	}
}

void Loader::loadMackeyGlassData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/mackey_glass.arff");
	
	size_t dims = 1;
	size_t offset = 30;
	size_t train_size = 300;
	size_t test_size = 600;
	
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		*y = raw[offset + i][1];
	}
}

void Loader::loadSunSpotData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/sunspots.arff");
	
	size_t dims = 1;
	size_t offset = 0;
	size_t granularity = 6;
	size_t train_size = 240;
	size_t test_size = 240;
	
	double scale = 0.01;
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		
		*y = 0.0;
		for(size_t j = 0; j < granularity; j++)
		{
			*y += scale * raw[offset + granularity * i + j][3];
		}
		*y /= granularity;
	}
}

void Loader::loadToyData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	size_t dims = 1;
	size_t train_size = 100;
	size_t test_size = 100;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	GRand r(time(NULL));
	size_t sines = 3;
	size_t randsPerDim = 2 * sines + 2;
	double rands[dims * randsPerDim];
	
	for(size_t i = 0; i < dims; i++)
	{
		for(size_t j = 0; j < sines; j++)
		{
			rands[randsPerDim * i + 2 * j] = r.normal() + 1.0;
			rands[randsPerDim * i + 2 * j + 1] = (r.next(6) + 6) * M_PI;
		}
		rands[randsPerDim * i + 2 * sines] = r.uniform() * 5;
		rands[randsPerDim * i + 1 * sines + 1] = r.uniform() * 0;
	}
	
	double *x, *y;
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = i / double(train_size);
		
		for(size_t j = 0; j < dims; j++)
		{
			y[j] = rands[randsPerDim * j + 2 * sines] * *x + rands[randsPerDim * j + 2 * sines + 1];
			for(size_t k = 0; k < sines; k++)
			{
				y[j] += rands[randsPerDim * j + 2 * k] * sin(rands[randsPerDim * j + 2 * k + 1] * *x);
			}
		}
	}
}

void Loader::loadToyData2(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	size_t dims = 1;
	size_t train_size = 300;
	size_t test_size = 300;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	double *x, *y;
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = i / double(train_size);
		*y = 2.0 * *x + 0.5 * sin(10.0 * M_PI * *x + tanh(60 * (*x - 0.5)) + 1);
	}
}

void Loader::loadTest01(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	size_t dims = 1;
	size_t train_size = 100;
	size_t test_size = 100;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	size_t sines = 3;
	size_t randsPerDim = 2 * sines + 2;
	double rands[dims * randsPerDim];
	
	for(size_t i = 0; i < dims; i++)
	{
		for(size_t j = 0; j < sines; j++)
		{
			rands[randsPerDim * i + 2 * j] = 1.0;
			rands[randsPerDim * i + 2 * j + 1] = (6.0 + 3.0 * j) * M_PI;
		}
		rands[randsPerDim * i + 2 * sines] = 2.5;
		rands[randsPerDim * i + 1 * sines + 1] = 0.0;
	}
	
	double *x, *y;
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = i / double(train_size);
		
		for(size_t j = 0; j < dims; j++)
		{
			y[j] = rands[randsPerDim * j + 2 * sines] * *x + rands[randsPerDim * j + 2 * sines + 1];
			for(size_t k = 0; k < sines; k++)
			{
				y[j] += rands[randsPerDim * j + 2 * k] * sin(rands[randsPerDim * j + 2 * k + 1] * *x);
			}
		}
	}
}

void Loader::loadTest02(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	size_t train_size = 100;
	size_t test_size = 100;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, 1);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, 1);
	
	size_t sines = 2;
	
	double *x, *y;
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = i / double(train_size);
		
		*y = 16.0;
		for(size_t k = 0; k < sines; k++)
		{
			*y -= sin((2 * M_PI * (k + 1)) * *x);
		}
	}
}

void Loader::loadTest04(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	size_t train_size = 128;
	size_t test_size = 256;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, 1);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, 1);
	
	size_t sines = 2;
	
	double *x, *y;
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = i / double(train_size);
		
		*y = 16.0 - 5 * *x;
		for(size_t k = 0; k < sines; k++)
		{
			*y -= sin((4.25 * M_PI * (k + 1)) * *x);
		}
	}
}

void Loader::loadNASDAQ(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix rawTrain, rawTest;
	rawTrain.loadArff("data/nasdaq_train.arff");
	rawTest.loadArff("data/nasdaq_test.arff");
	
	double *x, *y;
	
	trainFeat.resize(rawTrain.rows(), 1);
	trainLab.resize(rawTrain.rows(), 1);
	testFeat.resize(rawTest.rows(), 1);
	testLab.resize(rawTest.rows(), 1);
	
	for(size_t i = 0; i < rawTrain.rows() + rawTest.rows(); i++)
	{
		if(i < rawTrain.rows())
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - rawTrain.rows()];
			y = testLab[i - rawTrain.rows()];
		}
		
		*x = double(i) / rawTrain.rows();
		*y = i < rawTrain.rows() ? rawTrain[i][0] : rawTest[i - rawTrain.rows()][0];
		*y = *y - 3.25;
		*y = *y * 15;
	}
}

void Loader::loadAirPassengerData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/air_passengers.arff");
	
	size_t dims = 1;
	size_t offset = 0;
	size_t train_size = 72;
	size_t test_size = 72;
	
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	double log_10 = log(10);
	double vert_offset = 2;
	double scale = 10;//0.1;
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		*y = ((log(scale * raw[offset + i][0]) / log_10) - vert_offset);
	}
	
	trainLab.saveArff("out/train.arff");
	testLab.saveArff("out/test.arff");
}

void Loader::loadOzoneData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab)
{
	GMatrix raw;
	raw.loadArff("data/mhsets_monthly-ozone.arff");
	
	size_t dims = 1;
	size_t offset = 0;
	size_t train_size = 108;
	size_t test_size = 44;
	
	double *x, *y;
	
	trainFeat.resize(train_size, 1);
	trainLab.resize(train_size, dims);
	testFeat.resize(test_size, 1);
	testLab.resize(test_size, dims);
	
	for(size_t i = 0; i < train_size + test_size; i++)
	{
		if(i < train_size)
		{
			x = trainFeat[i];
			y = trainLab[i];
		}
		else
		{
			x = testFeat[i - train_size];
			y = testLab[i - train_size];
		}
		
		*x = double(i) / train_size;
		*y = log(raw[offset + i][0]) / log(10);
	}
	
	trainLab.saveArff("out/train.arff");
	testLab.saveArff("out/test.arff");
}
