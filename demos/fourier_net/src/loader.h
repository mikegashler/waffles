#ifndef LOADER_H
#define LOADER_H

#include <GClasses/GMatrix.h>
using namespace GClasses;

class Loader
{
	public:
		static void loadStockData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadPrecipitationData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadTemperatureData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadLaborData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadLaborData2(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadMackeyGlassData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadSunSpotData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadToyData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadToyData2(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		
		static void loadTest01(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadTest02(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadTest04(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		
		static void loadNASDAQ(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadAirPassengerData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
		static void loadOzoneData(GMatrix &trainFeat, GMatrix &trainLab, GMatrix &testFeat, GMatrix &testLab);
};

#endif
