/*
  The contents of this file are dedicated by all of its authors, including

    Michael S. Gashler,
    Eric Moyer,
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

#ifndef __GTOKENIZER_H__
#define __GTOKENIZER_H__

#include <istream>
#include <map>
#include "GBitTable.h"

namespace GClasses {


/// This class represents a set of characters.
class GCharSet
{
friend class GTokenizer;
protected:
	GBitTable m_bt;

public:
	/// szChars is an un-ordered set of characters (with no separator between
	/// them). The only special character is '-', which is used to indicate a
	/// range of characters if it is not the first character in the string.
	/// (So, if you want '-' in your set of characters, it should come first.)
	/// For example, the following string includes all letters: "a-zA-Z", and the
	/// following string includes all characters that might appear in a
	/// floating-point number: "-.,0-9e". (There is no way to include '\0' as
	/// a character in the set, since that character indicates the end of the
	/// string, but that is okay since '\0' should not occur in text files
	/// anyway, and this class is designed for parsing text files.)
	GCharSet(const char* szChars);

	/// Returns true iff c is in the character set
	bool find(char c) const;

	/// Returns true iff other is the same as this character set
	bool equals(const GCharSet& other) const;
};



#define GTOKENIZER_MAX_LOOKAHEAD 8

/// This is a simple tokenizer that reads a file, one token at-a-time.
/// Example usage:
///
/// GCharSet whitespace("\t\n\r ");
/// GCharSet alphanum("a-zA-Z0-9");
/// GCharSet float("-.,0-9e");
/// GCharSet commanewline(",\n");
/// GTokenizer tok(filename);
/// while(true)
/// {
/// 	tok.skip(whitespace);
/// 	if(!tok.has_more())
/// 		break;
/// 	char* mystr = tok.readWhile(alphanum);
/// 	tok.skip(commanewline);
/// }
class GTokenizer
{
protected:
	char m_q[GTOKENIZER_MAX_LOOKAHEAD]; // a look-ahead character queue for the stream
	size_t m_qPos; // the current head of the queue (it is a revolving queue)
	size_t m_qCount; // the number of characters in the queue
	char* m_pBufStart; // a buffer where the most-recently read token is stored
	char* m_pBufPos; // the current tail of the token buffer
	char* m_pBufEnd; // the end of the capacity of the token buffer
	std::istream* m_pStream; // the stream that is the source of the data
	size_t m_pos; // the position relative to the start of the stream
	size_t m_lineStart; // the position where this line started
	size_t m_line; // line number

public:
	/// Opens the specified filename.
	/// charSets is a class that inherits from GCharSetHolder
	GTokenizer(const char* szFilename);

	/// Uses the provided buffer of data. (If len is 0, then it
	/// will read until a null-terminator is found.)
	GTokenizer(const char* pFile, size_t len);

	virtual ~GTokenizer();

	/// Returns whether there is more data to be read
	bool has_more();

	/// Returns the next character in the stream. Returns '\0' if there are
	/// no more characters in the stream. (This could theoretically be ambiguous if the
	/// the next character in the stream is '\0', but presumably this class
	/// is mostly used for parsing text files, and that character should not
	/// occur in a text file.)
	char peek();

	/// Peek up to GTOKENIZER_MAX_LOOKAHEAD characters ahead. If n=0, returns the next character to be read.
	/// If n=1, retuns the second character ahead to be read, and so on.
	/// If n>=GTOKENIZER_MAX_LOOKAHEAD, throws an exception.
	char peek(size_t n);

	/// Appends a string to the current token (without modifying the file), and returns
	/// the full modified token.
	char* appendToToken(const char* string);

	/// Reads until the next character would be one of the specified delimeters.
	/// The delimeter character is not read. Throws an exception if fewer than
	/// minLen characters are read.
	/// The token returned by this method will have been copied into an
	/// internal buffer, null-terminated, and a pointer to that buffer is returned.
	char* readUntil(const GCharSet& delimeters, size_t minLen = 1, size_t maxLen = (size_t)-1);

	/// Reads until the next character would be one of the specified delimeters,
	/// and is not preceded by escapeChar.
	/// The token returned by this method will have been copied into an
	/// internal buffer, null-terminated, and a pointer to that buffer is returned.
	char* readUntil_escaped(char escapeChar, const GCharSet& delimeters);

	/// Reads while the character is one of the specified characters. Throws an
	/// exception if fewer than minLen characters are read.
	/// The token returned by this method will have been copied into an
	/// internal buffer, null-terminated, and a pointer to that buffer is returned.
	char* readWhile(const GCharSet& set, size_t minLen = 1);

	/// \brief Returns the next token defined by the given delimiters unless escaped or quoted with " or '.
	///
	/// Returns the next token delimited by the given delimiters.
	///
	/// The token may include delimiter characters if it is enclosed in quotes or
	/// the delimiters are escaped.
	///
	/// If the next token begins with single or double quotes, then the
	/// token will be delimited by the quotes. If a newline character or
	/// the end-of-file is encountered before the matching quote, then
	/// an exception is thrown. The quotation marks are included in
	/// the token.  The escape
	/// character is ignored inside quotes (unlike what would happen in
	/// C++).
	///
	/// If the first character of the token is not an apostrophe or quotation mark
	/// then it attempts to use the escape character to escape any special characters.
	/// That is, if the escape character appears, then the next character is
	/// interpreted to be part of the token. The
	/// escape character is consumed but not included in the token.
	/// Thus, if the input is (The \\\\rain\\\\ in \\\"spain\\\") (not
	/// including the parentheses) and the esapeChar is '\\', then the
	/// token read will be (The \\rain\\ in "spain").
	///
	/// No token may extend over multiple lines, thus the new-line
	/// character acts as an unescapable delimiter, no matter what set
	/// of delimiters is passed to the function.
	///
	///\param delimiters the set of delimiters used to separate tokens
	///
	///\param escapeChar the character that can be used to escape
	///                  delimiters when quoting is not active
	///
	///\return a pointer to an internal character buffer containing the
	///        null-terminated token
	char* readUntil_escaped_quoted(const GCharSet& delimiters, char escapeChar = '\\');

	/// Reads past any characters specified in the list of delimeters.
	/// If szDelimeters is NULL, then any characters <= ' ' are considered
	/// to be delimeters. (This method is similar to nextWhile, except that
	/// it does not buffer the characters it reads.)
	void skipWhile(const GCharSet& delimeters);

	/// Skip until the next character is one of the delimeters.
	/// (This method is the same as readUntil, except that it does not buffer what it reads.)
	void skipUntil(const GCharSet& delimeters);

	/// Advances past the next 'n' characters. (Stops if the end-of-file is reached.)
	void skip(size_t n);

	/// Reads past the specified string of characters. If the characters
	/// that are read from the file do not exactly match those in the string,
	/// an exception is thrown.
	void skipExact(const char* szString);

	/// Returns the previously-returned token, except with any of the specified characters
	/// trimmed off of both the beginning and end of the token. For example, this method could
	/// be used to convert "  tok  " to "tok".
	/// (Calling this method will not change the value returned by tokenLength.)
	char* trim(const GCharSet& set);

	/// Returns the previously-returned token, except with any characters not in the specified set removed.
	/// (Calling this method will not change the value returned by tokenLength.)
	char* filter(const GCharSet& set);

	/// Returns the current position in this stream
	size_t pos();

	/// Returns the current line number. (Begins at 1. Each time a '\n' is encountered,
	/// the line number is incremented. Mac line-endings do not increment the
	/// line number.)
	size_t line();

	/// Returns the current column index, which is the number of characters that have
	/// been read since the last newline character.
	size_t col();

	/// Returns the length of the last token that was returned.
	size_t tokenLength();

protected:
	/// Double the size of the token buffer.
	void growBuf();

	/// Returns the next character in the stream. If the next character is EOF, then it returns '\0'.
	char get();

	/// Read the next character into the token buffer.
	void bufferChar(char c);

	/// Add a '\0' to the end of the token buffer and return the token buffer.
	char* nullTerminate();
};

} // namespace GClasses

#endif // __GTOKENIZER_H__
