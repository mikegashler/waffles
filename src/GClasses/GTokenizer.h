/*
	Copyright (C) 2006, Mike Gashler

	This library is free software; you can redistribute it and/or
	modify it under the terms of the GNU Lesser General Public
	License as published by the Free Software Foundation; either
	version 2.1 of the License, or (at your option) any later version.

	see http://www.gnu.org/copyleft/lesser.html
*/

#ifndef __GTOKENIZER_H__
#define __GTOKENIZER_H__

#include <istream>

namespace GClasses {

/// This is a simple tokenizer that reads a file, one token at-a-time.
class GTokenizer
{
protected:
	char* m_pBufStart;
	char* m_pBufPos;
	char* m_pBufEnd;
	std::istream* m_pStream;
	size_t m_lineStart;
	size_t m_len;
	size_t m_line;

public:
	/// Opens the specified filename.
	GTokenizer(const char* szFilename);

	/// Uses the provided buffer of data. (If len is 0, then it
	/// will read until a null-terminator is found.)
	GTokenizer(const char* pFile, size_t len);
	~GTokenizer();

	/// Returns the next character in the stream. Returns '\0' if there are
	/// no more characters in the stream. (This could theoretically be ambiguous if the
	/// the next character in the stream is '\0', but presumably this class
	/// is mostly used for parsing text files, and that character should not
	/// occur in a text file.)
	char peek();

	/// This method skips past any characters in the string szDelimeters, then it
	/// reads a token until the next character would be a delimeter. If szDelimeters
	/// is NULL, then any characters <= ' ' are considered to be delimeters.
	/// A pointer to the buffered and null-terminated token is returned. NULL is
	/// returned if the end-of-file is reached. (Calling this method is
	/// the same as a call to skip, followed by a call to next, passing szDelimeters
	/// to both calls.)
	const char* nextTok(const char* szDelimeters = NULL);

	/// Reads until the next character is one of the specified delimeters.
	/// If szDelimeters is NULL, then any characters <= ' ' are considered
	/// to be delimeters. (The delimeter character is not read. Hence, it is typical
	/// to call skip before or after calling this method to advance past the delimeter.
	/// Repeatedly calling next will just stay in the same place and return
	/// empty tokens because it will never move past the delimeter. The nextTok
	/// method is provided for convenience to call both of these methods.)
	/// The token returned by this method will have been copied into an
	/// internal buffer, null-terminated, and a pointer to that buffer is returned.
	/// This method returns NULL if there are no more tokens in the file.
	const char* next(const char* szDelimeters = NULL);

	/// Returns the next token delimited by whitespace or a '{' character.
	/// If the next token begins
	/// with single or double quotes, then the token will be delimited by
	/// the quotes instead. If a newline character or the end-of-file is
	/// encountered before the matching quote, then an exception is thrown.
	/// The quotation marks are not included in the token, but they are
	/// consumed by the operation.
	const char* nextArg();

	/// Reads past any characters specified in the list of delimeters.
	/// If szDelimeters is NULL, then any characters <= ' ' are considered
	/// to be delimeters.
	void skip(const char* szDelimeters = NULL);

	/// Advances past the next 'n' characters. (Stops if the end-of-file is reached.)
	void advance(size_t n);

	/// Skip until the next character is one of the delimeters.
	/// (This method is the same as next, except that it does not buffer what it reads.)
	void skipTo(const char* szDelimeters = NULL);

	/// Reads past the specified string of characters. If the characters
	/// that are read from the file do not exactly match those in the string,
	/// an exception is thrown.
	void expect(const char* szString);

	/// Returns the current line number. (Begins at 1. Each time a '\n' is encountered,
	/// the line number is incremented. Mac line-endings do not increment the
	/// line number.)
	size_t line();

	/// Returns the current column index, which is the number of characters that have
	/// been read since the last newline character, plus 1.
	size_t col();

	/// Returns the number of remaining bytes to be read from the file.
	size_t remaining();

	/// Returns the length of the last token that was returned.
	size_t tokenLength();

	/// Returns the previously-returned token, except with whitespace trimmed
	/// off of both the beginning and end of the token. For example, if the last
	/// token that was returned was "  tok  ", then this will return "tok".
	/// (Calling this method will not change the value returned by tokenLength.)
	const char* trim();

protected:
	void growBuf();
};

} // namespace GClasses

#endif // __GTOKENIZER_H__
