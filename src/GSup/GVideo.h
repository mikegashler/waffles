/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GVIDEO_H__
#define __GVIDEO_H__

#include <vector>
#include <string.h>

namespace GClasses {

class GImage;

/// Represents a collection of images.
class GVideo
{
protected:
	std::vector<GImage*> m_frames;
	int m_nWidth, m_nHeight;

public:
	/// nWidth and nHeight specify the size of each frame in the video.
	/// (The video is initialized to contain no frames.)
	GVideo(int nWidth, int nHeight);
	~GVideo();

	/// Deletes all the frames and sets the size for future frames
	void setSize(int width, int height);

	/// Returns the width of each frame in the video
	int width() const { return m_nWidth; }

	/// Returns the height of each frame in the video
	int height() const { return m_nHeight; }

	/// Returns the number of frames
	int frameCount();

	/// Returns the specified frame
	GImage* frame(int index);

	/// Appends a blank frame to the end of the video
	void addBlankFrame();

	/// Loads a PNG file and appends it as a frame to the video. The dimensions
	/// of the PNG file must match the dimensions of the video frames.
	bool loadFrame(const char* szFilename);

	/// Makes this video into a gradient magnitude video. If bForDisplay is true,
	/// it will put the magnitude in each channel so that the resultant video
	/// will make visual sense.
	void makeGradientMagnitudeVideo(GVideo* pVideo, bool bForDisplay);
};

} // namespace GClasses

#endif // __GVIDEO_H__
