/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GWIDGETS_H__
#define __GWIDGETS_H__

#include "GRect.h"
#include "GImage.h"
#include <string>
#include <vector>

namespace GClasses {

class GWidgetAtomic;
class GWidgetBulletGroup;
class GWidgetBulletHole;
class GWidgetCanvas;
class GWidgetCheckBox;
class GWidgetGroup;
class GWidget;
class GWidgetTextButton;
class GWidgetTextTab;
class GWidgetImageButton;
class GWidgetVCRButton;
class GWidgetHorizScrollBar;
class GWidgetHorizSlider;
class GWidgetVertScrollBar;
class GWidgetTextLabel;
class GWidgetFileSystemBrowser;
class GWidgetSliderTab;
class GWidgetTextBox;
class GWidgetPolarLineGraph;
class GWidgetPolarBarGraph;
class GWidgetVertSlider;
class GWave;

class GWidgetCommon
{
protected:
	GImage m_bufferImage;

public:
	GWidgetCommon();
	~GWidgetCommon();

	void drawButtonText(GImage* pImage, int x, int y, int w, int h, const char* text, bool pressed);
	void drawLabelText(GImage* pImage, int x, int y, int w, int h, const char* text, float fontSize, bool alignLeft, unsigned int c);
	void drawHorizCurvedOutSurface(GImage* pImage, int x, int y, int w, int h, unsigned int col = 0xff0000ff);
	void drawHorizCurvedInSurface(GImage* pImage, int x, int y, int w, int h, unsigned int col = 0xff0000ff);
	void drawVertCurvedOutSurface(GImage* pImage, int x, int y, int w, int h, unsigned int col = 0xff0000ff);
	void drawVertCurvedInSurface(GImage* pImage, int x, int y, int w, int h, unsigned int col = 0xff0000ff);
	void drawCursor(GImage* pImage, int x, int y, int w, int h);

	/// This method lets you draw a widget clipped by some rect. Note that
	/// it uses a persistent buffer image to facilitate this. It doesn't
	/// clear the image, so if your widget doesn't draw over its entire
	/// rect, you'll see garbage from previous uses
	void drawClipped(GImage* pCanvas, int x, int y, GWidget* pWidget, GRect* pClipRect);
};




/// The base class of all GUI widgets
class GWidget
{
friend class GWidgetGroup;
friend class GWidgetGrid;
public:
	enum WidgetType
	{
		Animation,
		BulletGroup,
		BulletHole,
		Canvas,
		CheckBox,
		Custom,
		Dialog,
		FileSystemBrowser,
		Grid,
		GroupBox,
		HorizSlider,
		HScrollBar,
		ImageLabel,
		PolarChart,
		ProgressBar,
		SliderTab,
		TextBox,
		TextButton,
		TextLabel,
		TextTab,
		VCRButton,
		VertSlider,
		VScrollBar,
		Wave,
	};

protected:
	GRect m_rect;
	GWidgetGroup* m_pParent;
	GWidgetCommon* m_common;
	int m_nID; // for use by the owning parent

public:
	GWidget(GWidgetGroup* m_pParent, int x, int y, int w, int h);
	virtual ~GWidget();

	virtual WidgetType type() = 0;
	virtual bool isAtomic() = 0;

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y) = 0;
	void setPos(int x, int y);
	GRect* rect() { return &m_rect; }
	int id() { return m_nID; }
	GWidgetGroup* parent() { return m_pParent; }
};





/// The base class of all atomic widgets (widgets that are not composed of other widgets).
class GWidgetAtomic : public GWidget
{
friend class GWidgetDialog;
public:
	GWidgetAtomic(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetAtomic();

	virtual bool isAtomic() { return true; }
	virtual void onChar(char c);
	virtual void onSpecialKey(int key);
	virtual void onMouseMove(int dx, int dy);
	virtual void onGetFocus() {}
	virtual void onLoseFocus() {}
	virtual bool isClickable() { return true; }

protected:
	virtual void grab(int button, int x, int y) {}
	virtual void release(int button) {}
};






/// The base class of all widgets that are composed of other widgets
class GWidgetGroup : public GWidget
{
friend class GWidget;
friend class GWidgetAtomic;
protected:
	std::vector<GWidget*> m_widgets;
	std::vector<GWidget*> m_dirtyChildren;
	std::vector<unsigned int> m_dirtyBits;

public:
	GWidgetGroup(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetGroup();

	virtual bool isAtomic() { return false; }
	virtual GWidgetAtomic* findAtomicWidget(int x, int y);
	virtual void onDestroyWidget(GWidget* pWidget);
	int childWidgetCount();
	GWidget* childWidget(int n);
	virtual void tattle(GWidget* pChild);

	virtual void onPushTextButton(GWidgetTextButton* pButton)
	{
		if(m_pParent)
			m_pParent->onPushTextButton(pButton);
	}

	virtual void onReleaseTextButton(GWidgetTextButton* pButton)
	{
		if(m_pParent)
			m_pParent->onReleaseTextButton(pButton);
	}

	virtual void onReleaseImageButton(GWidgetImageButton* pButton)
	{
		if(m_pParent)
			m_pParent->onReleaseImageButton(pButton);
	}

	virtual void onPushVCRButton(GWidgetVCRButton* pButton)
	{
		if(m_pParent)
			m_pParent->onPushVCRButton(pButton);
	}

	virtual void onHorizScroll(GWidgetHorizScrollBar* pScrollBar)
	{
		if(m_pParent)
			m_pParent->onHorizScroll(pScrollBar);
	}

	virtual void onVertScroll(GWidgetVertScrollBar* pScrollBar)
	{
		if(m_pParent)
			m_pParent->onVertScroll(pScrollBar);
	}

	virtual void onClickTextLabel(GWidgetTextLabel* pLabel)
	{
		if(m_pParent)
			m_pParent->onClickTextLabel(pLabel);
	}

	virtual void onSelectFilename(GWidgetFileSystemBrowser* pBrowser, const char* szFilename)
	{
		if(m_pParent)
			m_pParent->onSelectFilename(pBrowser, szFilename);
	}

	virtual void onTextBoxTextChanged(GWidgetTextBox* pTextBox)
	{
		if(m_pParent)
			m_pParent->onTextBoxTextChanged(pTextBox);
	}

	virtual void onTextBoxPressEnter(GWidgetTextBox* pTextBox)
	{
		if(m_pParent)
			m_pParent->onTextBoxPressEnter(pTextBox);
	}

	virtual void onChar(char c)
	{
		if(m_pParent)
			m_pParent->onChar(c);
	}

	virtual void onSpecialKey(int key)
	{
		if(m_pParent)
			m_pParent->onSpecialKey(key);
	}

	virtual void onClickTab(GWidgetSliderTab* pTab)
	{
		if(m_pParent)
			m_pParent->onClickTab(pTab);
	}

	virtual void onSlideTab(GWidgetSliderTab* pTab, int dx, int dy)
	{
		if(m_pParent)
			m_pParent->onSlideTab(pTab, dx, dy);
	}

	virtual void onHorizSliderMove(GWidgetHorizSlider* pSlider)
	{
		if(m_pParent)
			m_pParent->onHorizSliderMove(pSlider);
	}

	virtual void onVertSliderMove(GWidgetVertSlider* pSlider)
	{
		if(m_pParent)
			m_pParent->onVertSliderMove(pSlider);
	}

	virtual void onCanvasMouseDown(GWidgetCanvas* pCanvas, int button, int x, int y)
	{
		if(m_pParent)
			m_pParent->onCanvasMouseDown(pCanvas, button, x, y);
	}

	virtual void onCanvasMouseUp(GWidgetCanvas* pCanvas, int button, int x, int y)
	{
		if(m_pParent)
			m_pParent->onCanvasMouseUp(pCanvas, button, x, y);
	}

	virtual void onCanvasMouseMove(GWidgetCanvas* pCanvas, int x, int y, bool bPressed)
	{
		if(m_pParent)
			m_pParent->onCanvasMouseMove(pCanvas, x, y, bPressed);
	}

	virtual void onSelectTextTab(GWidgetTextTab* pTab)
	{
		if(m_pParent)
			m_pParent->onSelectTextTab(pTab);
	}

	virtual void onCheckBulletHole(GWidgetBulletHole* pBullet)
	{
		if(m_pParent)
			m_pParent->onCheckBulletHole(pBullet);
	}

	virtual void onChangeCheckBox(GWidgetCheckBox* pCheckBox)
	{
		if(m_pParent)
			m_pParent->onChangeCheckBox(pCheckBox);
	}

	virtual void onCustomEvent(GWidget* pWidget)
	{
		if(m_pParent)
			m_pParent->onCustomEvent(pWidget);
	}

protected:
	void addWidget(GWidget* pWidget);
	//void UpdateChildOntoMyCanvas(GWidget* pChild, unsigned int cBackground);
	bool getDirtyBit(int nBit);
	void setDirtyBit(int nBit, bool bValue);
	void setClean();
};







/// A form or dialog
class GWidgetDialog : public GWidgetGroup
{
protected:
	GImage m_image;
	GWidgetAtomic* m_pGrabbedWidget;
	GWidgetAtomic* m_pFocusWidget;
	unsigned int m_cBackground;
	int m_prevMouseX;
	int m_prevMouseY;
	bool m_bRunning;

public:
	GWidgetDialog(int w, int h, unsigned int cBackground);
	virtual ~GWidgetDialog();

	virtual WidgetType type() { return Dialog; }

	/// Sets the background image
	void setBackgroundImage(GImage* pImage);

	/// Returns the widget that the mouse is currently grabbing
	GWidgetAtomic* grabbedWidget() { return m_pGrabbedWidget; }

	/// Returns the widget that currently has focus
	GWidgetAtomic* focusWidget() { return m_pFocusWidget; }

	/// Sets the widget with focus
	void setFocusWidget(GWidgetAtomic* pWidget);

	/// You should call this when the user presses a mouse button
	virtual void pressButton(int button, int x, int y);

	/// You should call this method when the mouse releases (un-clicks)
	virtual void releaseButton(int button);

	virtual void onDestroyWidget(GWidget* pWidget);

	/// You should call this method when the user presses a key
	virtual void handleChar(char c);

	/// You should call this method when the user presses a special key
	virtual void handleSpecialKey(int key);

	/// You should call this method when the user moves the mouse
	virtual bool handleMousePos(int x, int y);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Updates everything that needs to be updated, and makes an image
	/// of the dialog in its current state so you can blit it to the screen
	GImage* image();

	/// Returns the background color
	unsigned int backgroundColor() { return m_cBackground; }

	/// This method is used when running the dialog as a popup modal dialog.
	/// The flag is initialized to true. It is the controller's job to check
	/// this flag and close the dialog when it is set to false.
	bool* runningFlag() { return &m_bRunning; }

	/// Sets m_bRunning to false. It's the controller's job to watch this
	/// flag and do something about it.
	void close() { m_bRunning = false; }

protected:
	/// This is called by pressAt when the user clicks on an atomic widget
	void grabWidget(GWidgetAtomic* pWidget, int button, int mouseX, int mouseY);
};




/// A button with text on it
class GWidgetTextButton : public GWidgetAtomic
{
protected:
	std::string m_text;
	bool m_pressed, m_holding;
	int m_pressedX, m_pressedY;
	unsigned int m_color;

public:
	GWidgetTextButton(GWidgetGroup* pParent, int x, int y, int w, int h, const char* szText);
	virtual ~GWidgetTextButton();

	virtual WidgetType type() { return TextButton; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// sets the button text
	void setText(const char* szText);

	/// sets the button color
	void setColor(unsigned int c);

	/// returns "true" if the button is currently pressed
	bool isPressed() { return m_pressed; }

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
	virtual void onMouseMove(int dx, int dy);
};





/// Represents a tab (like for tabbed menus, etc.)
class GWidgetTextTab : public GWidgetAtomic
{
protected:
	std::string m_text;
	unsigned int m_cBackground;
	bool m_selected;

public:
	GWidgetTextTab(GWidgetGroup* pParent, int x, int y, int w, int h, const char* szText, unsigned int cBackground = 0xff6600aa);
	virtual ~GWidgetTextTab();

	virtual WidgetType type() { return TextTab; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// sets the tab text
	void setText(const char* szText);

	/// returns "true" if the tab is currently selected
	bool isSelected() { return m_selected; }

	/// sets the tab as selected or not selected
	void setSelected(bool selected);

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};





/// A button with an image on it. The left half of the image is the
/// unpressed image and the right half is the pressed image.
class GWidgetImageButton : public GWidgetAtomic
{
protected:
	GImage m_image;
	bool m_pressed, m_holding;
	int m_pressedX, m_pressedY;

public:
	GWidgetImageButton(GWidgetGroup* pParent, int x, int y, GImage* pImage);
	virtual ~GWidgetImageButton();

	virtual WidgetType type() { return TextButton; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Returns true if the button is currently pressed
	bool isPressed() { return m_pressed; }

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
	virtual void onMouseMove(int dx, int dy);
};





/// An image with multiple frames
class GWidgetAnimation : public GWidgetAtomic
{
protected:
	GImage m_image;
	int m_nFrames;
	int m_nFrame;

public:
	GWidgetAnimation(GWidgetGroup* pParent, int x, int y, GImage* pImage, int nFrames);
	virtual ~GWidgetAnimation();

	virtual WidgetType type() { return Animation; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets the current animation frame
	void setFrame(int nFrame);
};





/// A text label
class GWidgetTextLabel : public GWidgetAtomic
{
protected:
	std::string m_text;
	bool m_alignLeft;
	float m_fontSize;
	unsigned int m_cBackground;
	unsigned int m_cForeground;
	bool m_bGrabbed;

public:
	GWidgetTextLabel(GWidgetGroup* pParent, int x, int y, int w, int h, const char* szText, unsigned int c = 0xffffffff, unsigned int background = 0x00000000, float fontSize = 1.0f);
	virtual ~GWidgetTextLabel();

	virtual WidgetType type() { return TextLabel; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Returns the label text
	const std::string& text() { return m_text; }

	/// Sets the size of the font
	void setFontSize(float f) { m_fontSize = f; }

	/// Sets the label text
	void setText(const char* szText);

	/// Sets the text color
	void setForegroundColor(unsigned int c);

	/// Gets the text color
	unsigned int foregroundColor() { return m_cForeground; }

	/// The default background color is transparent. If you want an opaque
	/// or semi-opaque background then you should call this method.
	void setBackgroundColor(unsigned int c);

	/// Gets the text color
	unsigned int backgroundColor() { return m_cBackground; }

	/// Specifies whether the text is left-justified (true) or right-justified (false)
	void setAlignLeft(bool bAlignLeft);

	/// Insert newline characters into the current text at
	/// good locations to make the text wrap within its rect
	void wrap();

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};



class GWidgetImageLabel : public GWidgetAtomic
{
protected:
	GImage m_image;

public:
	GWidgetImageLabel(GWidgetGroup* pParent, int x, int y, const char* szHexPngFile);
	virtual ~GWidgetImageLabel();

	virtual WidgetType type() { return ImageLabel; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);
};




/// This just draws a rectangular box.
class GWidgetGroupBox : public GWidgetAtomic
{
protected:
	unsigned int m_cLight;
	unsigned int m_cShadow;

public:
	GWidgetGroupBox(GWidgetGroup* pParent, int x, int y, int w, int h, unsigned int cLight = 0xffc0c0c0, unsigned int cShadow = 0xff404040);
	virtual ~GWidgetGroupBox();

	virtual WidgetType type() { return GroupBox; }

	virtual bool isClickable() { return false; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets the light color
	void setLightColor(unsigned int c);

	/// Sets the shadow color
	void setShadowColor(unsigned int c);
};




/// A button with a common icon on it
class GWidgetVCRButton : public GWidgetAtomic
{
public:
	enum VCR_Type
	{
		ArrowLeft,
		ArrowRight, // Play
		ArrowUp,
		ArrowDown,
		Square,  // Stop
	};

protected:
	VCR_Type m_eType;
	bool m_pressed;

public:
	GWidgetVCRButton(GWidgetGroup* pParent, int x, int y, int w, int h, VCR_Type eType);
	virtual ~GWidgetVCRButton();

	virtual WidgetType type() { return VCRButton; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets the button image
	void setType(VCR_Type eType);

	/// Returns true if the button is currently pressed
	bool isPressed() { return m_pressed; }

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
	void drawIcon(GImage* pCanvas, int nHorizOfs, int nVertOfs);
};





/// Automatically determines wether to be horizontal or vertical
/// based on dimensions. Progress ranges from 0 to 1, or from 0 to -1 if
/// you want it to go the other way.
class GWidgetProgressBar : public GWidgetAtomic
{
protected:
	float m_fProgress;

public:
	GWidgetProgressBar(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetProgressBar();

	virtual WidgetType type() { return ProgressBar; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets the current position of the progress bar (from 0 to 1)
	void setProgress(float fProgress);

	/// Gets the current position of the progress bar (from 0 to 1)
	float progress() { return m_fProgress; }
};






class GWidgetCheckBox : public GWidgetAtomic
{
protected:
	bool m_checked;

public:
	GWidgetCheckBox(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetCheckBox();

	virtual WidgetType type() { return CheckBox; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets whether or not the box has an "X" in it
	void setChecked(bool checked);

	/// Returns true if the box is currently checked
	bool isChecked() { return m_checked; }

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};




/// The easiest way to do bullets is to use the GWidgetBulletGroup class,
/// but if you really want to manage individual bullets yourself, you can
/// use this class to do it.
class GWidgetBulletHole : public GWidgetAtomic
{
protected:
	bool m_checked;

public:
	GWidgetBulletHole(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetBulletHole();

	virtual WidgetType type() { return BulletHole; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Puts a dot inside this bullet hole
	void setChecked(bool checked);

	/// Returns true if there is a dot in this bullet hole
	bool isChecked() { return m_checked; }

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};




/// This creates a whole group of bullets arranged either horizontally
/// or vertically at regular intervals.
class GWidgetBulletGroup : public GWidgetGroup
{
protected:
	int m_nSelection;

public:
	/// "w" and "h" are the width and height of a single bullet hole. "count" is the number
	/// of bullet holes in the group. "interval" is the distance from the center of one
	/// bullet hole to the center of the next one. If "vertical" is true, they are arranged
	/// vertically starting at (x,y) downward. If "vertical" is false, they are arranged
	/// horizontally from (x,y) to the right.
	GWidgetBulletGroup(GWidgetGroup* pParent, int x, int y, int w, int h, int count, int interval, bool vertical);
	virtual ~GWidgetBulletGroup();

	virtual WidgetType type() { return BulletGroup; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Sets which bullet hole has a dot in it
	void setSelection(int n);

	/// Returns the index of the bullet hole with the dot in it
	int selection() { return m_nSelection; }

protected:
	virtual void onCheckBulletHole(GWidgetBulletHole* pBullet);
	virtual void tattle(GWidget* pChild);
};




/// This widget is not meant to be used by itself. It creates one of the parts of a scroll bar or slider bar.
class GWidgetSliderTab : public GWidgetAtomic
{
public:
	enum Style
	{
		ScrollBarTab,
		ScrollBarArea,
		SliderNub,
		SliderArea,
	};

protected:
	bool m_vertical;
	Style m_eStyle;

public:
	GWidgetSliderTab(GWidgetGroup* pParent, int x, int y, int w, int h, bool vertical, Style eStyle);
	virtual ~GWidgetSliderTab();

	virtual WidgetType type() { return SliderTab; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Changes the size of this widget
	void setSize(int w, int h);

protected:
	virtual void onMouseMove(int dx, int dy);
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};






/// Makes a horizontal scroll bar
class GWidgetHorizScrollBar : public GWidgetGroup
{
protected:
	int m_nViewSize;
	int m_nModelSize;
	int m_nPos;
	GWidgetVCRButton* m_pLeftButton;
	GWidgetVCRButton* m_pRightButton;
	GWidgetSliderTab* m_pLeftTab;
	GWidgetSliderTab* m_pTab;
	GWidgetSliderTab* m_pRightTab;

public:
	GWidgetHorizScrollBar(GWidgetGroup* pParent, int x, int y, int w, int h, int nViewSize, int nModelSize);
	virtual ~GWidgetHorizScrollBar();

	virtual WidgetType type() { return HScrollBar; }

	/// Returns the current view size
	int viewSize() { return m_nViewSize; }

	/// Returns the current model size
	int modelSize() { return m_nModelSize; }

	/// Sets the size of the view area that this scroll bar represents
	/// (The size of the sliding tab is determined by the ratio of the
	/// size of the view over the size of the model)
	void setViewSize(int n);

	/// Sets the size of the model that this scroll bar represents
	/// (The size of the sliding tab is determined by the ratio of the
	/// size of the view over the size of the model)
	void setModelSize(int n);

	/// Gets the current scroll position of this scroll bar
	int pos() { return m_nPos; }

	/// Sets the current scroll position
	void setPos(int n);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

protected:
	virtual void onPushVCRButton(GWidgetVCRButton* pButton);
	virtual void onSlideTab(GWidgetSliderTab* pTab, int dx, int dy);
	virtual void onClickTab(GWidgetSliderTab* pTab);
	int buttonWidth();
};





/// Makes a vertical scroll bar
class GWidgetVertScrollBar : public GWidgetGroup
{
protected:
	int m_nViewSize;
	int m_nModelSize;
	int m_nPos;
	GWidgetVCRButton* m_pUpButton;
	GWidgetVCRButton* m_pDownButton;
	GWidgetSliderTab* m_pAboveTab;
	GWidgetSliderTab* m_pTab;
	GWidgetSliderTab* m_pBelowTab;

public:
	GWidgetVertScrollBar(GWidgetGroup* pParent, int x, int y, int w, int h, int nViewSize, int nModelSize);
	virtual ~GWidgetVertScrollBar();

	virtual WidgetType type() { return VScrollBar; }

	/// Returns the current view size
	int viewSize() { return m_nViewSize; }

	/// Returns the current model size
	int modelSize() { return m_nModelSize; }

	/// Sets the size of the view area that this scroll bar represents
	/// (The size of the sliding tab is determined by the ratio of the
	/// size of the view over the size of the model)
	void setViewSize(int n);

	/// Sets the size of the model that this scroll bar represents
	/// (The size of the sliding tab is determined by the ratio of the
	/// size of the view over the size of the model)
	void setModelSize(int n);

	/// Gets the current scroll position of this scroll bar
	int pos() { return m_nPos; }

	/// Sets the current scroll position
	void setPos(int n);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

protected:
	virtual void onPushVCRButton(GWidgetVCRButton* pButton);
	virtual void onSlideTab(GWidgetSliderTab* pTab, int dx, int dy);
	virtual void onClickTab(GWidgetSliderTab* pTab);
	int buttonHeight();
};






/// This is a box in which the user can enter text
class GWidgetTextBox : public GWidgetAtomic
{
protected:
	std::string m_text;
	bool m_bGotFocus;
	bool m_bPassword;
	int m_nAnchorPos;
	int m_nCursorPos;
	int m_nMouseDelta;
	unsigned int m_cBackground;

public:
	GWidgetTextBox(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetTextBox();

	virtual WidgetType type() { return TextBox; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	/// Returns the current contents of this text box
	const std::string& text() { return m_text; }

	/// Sets the text in this text box
	void setText(const char* szText);
	void setText(int nValue);
	void setText(double dValue);

	/// Sets the cursor position and selection
	void SetSelection(int anchorPos, int cursorPos);

	virtual void onChar(char c);
	virtual void onSpecialKey(int key);

	/// Sets whether or not it should display a bunch of '#'s instead
	/// of the current text
	void setPassword() { m_bPassword = true; }

	void setColor(unsigned int c);

protected:
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
	virtual void onGetFocus();
	virtual void onLoseFocus();
	virtual void onMouseMove(int dx, int dy);
};




class GWidgetGrid : public GWidgetGroup
{
protected:
	std::vector<GWidget**> m_rows;
	int m_nColumns;
	int m_nRowHeight;
	int m_nHeaderHeight;
	GWidget** m_pColumnHeaders;
	int* m_nColumnWidths;
	GWidgetVertScrollBar* m_pVertScrollBar;
	GWidgetHorizScrollBar* m_pHorizScrollBar;
	unsigned int m_cBackground;

public:
	GWidgetGrid(GWidgetGroup* pParent, int nColumns, int x, int y, int w, int h, unsigned int cBackground = 0xff000000);
	virtual ~GWidgetGrid();

	virtual WidgetType type() { return Grid; }
	virtual GWidgetAtomic* findAtomicWidget(int x, int y);
	int rowHeight() { return m_nRowHeight; }
	void setRowHeight(int n);
	int headerHeight() { return m_nHeaderHeight; }
	void setHeaderHeight(int n);
	int hScrollPos() { return m_pHorizScrollBar->pos(); }
	int vScrollPos() { return m_pVertScrollBar->pos(); }
	void setHScrollPos(int n);
	void setVScrollPos(int n);

	/// Adds an empty row to the grid
	void addBlankRow();

	/// Sets the number of columns (preserving any widgets that still fit in the new grid)
	void setColumnCount(int n);

	/// Returns the number of columns
	int columnCount() { return m_nColumns; }

	/// Sets the widget in a column header
	GWidget* columnHeader(int col);

	/// Gets the widget in a column header
	void setColumnHeader(int col, GWidget* pWidget);

	/// Gets the width of a column
	int columnWidth(int col);

	/// Sets the width of a column
	void setColumnWidth(int col, int nWidth);

	/// Returns the number of rows
	int rowCount();

	/// Gets the widget in the specified cell
	GWidget* widget(int col, int row);

	/// Sets the widget in the specified cell
	void setWidget(int col, int row, GWidget* pWidget);

	/// Deletes all the rows and all the widgets in them
	void flushItems(bool deleteWidgets = true);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	virtual void tattle(GWidget* pChild);

protected:
	virtual void onVertScroll(GWidgetVertScrollBar* pScrollBar);
	virtual void onHorizScroll(GWidgetHorizScrollBar* pScrollBar);
};





class GWidgetFileSystemBrowser : public GWidgetGroup
{
protected:
	std::string m_path;
	GWidgetTextLabel* m_pPath;
	GWidgetGrid* m_pFiles;
	char* m_szExtensions;
	bool m_bFileListDirty;

public:
	/// szExtension should be NULL if you want to allow all extensions
	GWidgetFileSystemBrowser(GWidgetGroup* pParent, int x, int y, int w, int h, const char* szExtensions);
	virtual ~GWidgetFileSystemBrowser();

	virtual WidgetType type() { return FileSystemBrowser; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

	void setFileListDirty() { m_bFileListDirty = true; tattle(NULL); }

protected:
	virtual void dirFoldersAndFiles(std::string* pOutDir, std::vector<std::string>* pOutFolders, std::vector<std::string>* pOutFiles);
	virtual void onClickTextLabel(GWidgetTextLabel* pLabel);
	void reloadFileList();
	void addFilename(bool bDir, const char* szFilename);
};



class GWidgetHorizSlider : public GWidgetGroup
{
protected:
	bool m_dirty;
	float m_fPos;
	GWidgetSliderTab* m_pLeftTab;
	GWidgetSliderTab* m_pTab;
	GWidgetSliderTab* m_pRightTab;

public:
	GWidgetHorizSlider(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetHorizSlider();

	virtual WidgetType type() { return HorizSlider; }
	float pos() { return m_fPos; }
	void setPos(float f);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

protected:
	virtual void onSlideTab(GWidgetSliderTab* pTab, int dx, int dy);
	virtual void onClickTab(GWidgetSliderTab* pTab);
};



class GWidgetVertSlider : public GWidgetGroup
{
protected:
	bool m_dirty;
	float m_fPos;
	GWidgetSliderTab* m_pAboveTab;
	GWidgetSliderTab* m_pTab;
	GWidgetSliderTab* m_pBelowTab;

public:
	GWidgetVertSlider(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetVertSlider();

	virtual WidgetType type() { return VertSlider; }
	float pos() { return m_fPos; }
	void setPos(float f);

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);

protected:
	virtual void onSlideTab(GWidgetSliderTab* pTab, int dx, int dy);
	virtual void onClickTab(GWidgetSliderTab* pTab);
};



/// A painting canvas
class GWidgetCanvas : public GWidgetAtomic
{
protected:
	GImage* m_pImageIn;
	GImage* m_pSelectionMask;
	float m_fZoom;
	int m_nHScroll;
	int m_nVScroll;
	bool m_pressed;
	int m_nMouseX, m_nMouseY;

public:
	GWidgetCanvas(GWidgetGroup* pParent, int x, int y, int w, int h, GImage* pImage);
	virtual ~GWidgetCanvas();

	virtual WidgetType type() { return Canvas; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);
	void setDirty();

	float zoom() { return m_fZoom; }
	void setZoom(float f);
	void zoomToFit();
	void setHorizScroll(int x);
	void setVertScroll(int y);

	void setImage(GImage* pImage);

	/// Only the alpha channel of the mask is used. The other values should be
	/// constant, or else the selection border will cut through the selection.
	/// If you change what is selected, you should call this method again, even
	/// though it may still be the same mask image.
	void setSelectionMask(GImage* pMask);

protected:
	virtual void onMouseMove(int dx, int dy);
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};



class GWidgetWave : public GWidgetAtomic
{
protected:
	GWave* m_pWave;
	int m_pos, m_width, m_selectionPos, m_selectionWidth;

public:
	GWidgetWave(GWidgetGroup* pParent, int x, int y, int w, int h);
	virtual ~GWidgetWave();

	virtual WidgetType type() { return Wave; }

	/// Draws this widget on pCanvas at (x,y)
	virtual void draw(GImage* pCanvas, int x, int y);
	void setDirty();

	void setWave(GWave* pWave);
	int pos() { return m_pos; }
	int width() { return m_width; }
	void setRange(int pos, int width);

protected:
	virtual void onMouseMove(int dx, int dy);
	virtual void grab(int button, int x, int y);
	virtual void release(int button);
};



} // namespace GClasses

#endif // __GWIDGETS_H__
