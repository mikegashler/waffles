#!/bin/bash
set -u -e


# Generate the waffles_learn doc
cat <<ENDTXT > command/learn.html
<html><body>
<table border="0" cellpadding="0" cellspacing="0" width="980" bgcolor="#f4f0e5">
<tr><td background="../images/bar.png"><br>
</td></tr><tr><td>
<a href="../docs.html">Back to the docs page</a><br>

<br>
<a href="../tutorial/wizard.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="transform.html">Next</a>



<h2>waffles_learn</h2>
<p>
	A command-line tool that wraps supervised and semi-supervised learning algorithms.
	Here's the usage information:</p>
<pre>
ENDTXT
waffles_learn usage | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' >> command/learn.html
cat <<ENDTXT >> command/learn.html
</pre>



<br>
<a href="../tutorial/wizard.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="transform.html">Next</a>

<br><br><a href="../docs.html">Back to the docs page</a>
</td></tr><tr><td background="../images/bar.png"><br>
</td></tr></table>
</body></html>
ENDTXT



# Generate the waffles_transform doc
cat <<ENDTXT > command/transform.html
<html><body>
<table border="0" cellpadding="0" cellspacing="0" width="980" bgcolor="#f4f0e5">
<tr><td background="../images/bar.png"><br>
</td></tr><tr><td>
<a href="../docs.html">Back to the docs page</a><br>

<br>
<a href="learn.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="plot.html">Next</a>



<h2>waffles_transform</h2>
<p>
	A command-line tool for transforming datasets. It contains import/export functionality,
	unsupervised algorithms, and other useful transforms that you may wish to perform on a dataset.
	Here's the usage information:</p>
<pre>
ENDTXT
waffles_transform usage | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' >> command/transform.html
cat <<ENDTXT >> command/transform.html
</pre>



<br>
<a href="learn.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="plot.html">Next</a>

<br><br><a href="../docs.html">Back to the docs page</a>
</td></tr><tr><td background="../images/bar.png"><br>
</td></tr></table>
</body></html>
ENDTXT


# Generate the waffles_plot doc
cat <<ENDTXT > command/plot.html
<html><body>
<table border="0" cellpadding="0" cellspacing="0" width="980" bgcolor="#f4f0e5">
<tr><td background="../images/bar.png"><br>
</td></tr><tr><td>
<a href="../docs.html">Back to the docs page</a><br>

<br>
<a href="transform.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="generate.html">Next</a>



<h2>waffles_plot</h2>
<p>
	A command-line tool for plotting and visualizing datasets.
	Here's the usage information:</p>
<pre>
ENDTXT
waffles_plot usage | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' >> command/plot.html
cat <<ENDTXT >> command/plot.html
</pre>



<br>
<a href="transform.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="generate.html">Next</a>

<br><br><a href="../docs.html">Back to the docs page</a>
</td></tr><tr><td background="../images/bar.png"><br>
</td></tr></table>
</body></html>
ENDTXT


# Generate the waffles_generate doc
cat <<ENDTXT > command/generate.html
<html><body>
<table border="0" cellpadding="0" cellspacing="0" width="980" bgcolor="#f4f0e5">
<tr><td background="../images/bar.png"><br>
</td></tr><tr><td>
<a href="../docs.html">Back to the docs page</a><br>

<br>
<a href="plot.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="recommend.html">Next</a>




<h2>waffles_generate</h2>
<p>
	A command-line tool to help generate various types of data.
	(Most of the datasets it generates are for testing manifold learning
	algorithms. I add them as I need them.) Here's the usage information:</p>
<pre>
ENDTXT
waffles_generate usage | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' >> command/generate.html
cat <<ENDTXT >> command/generate.html
</pre>



<br>
<a href="plot.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="recommend.html">Next</a>

<br><br><a href="../docs.html">Back to the docs page</a>
</td></tr><tr><td background="../images/bar.png"><br>
</td></tr></table>
</body></html>
ENDTXT


# Generate the waffles_recommend doc
cat <<ENDTXT > command/recommend.html
<html><body>
<table border="0" cellpadding="0" cellspacing="0" width="980" bgcolor="#f4f0e5">
<tr><td background="../images/bar.png"><br>
</td></tr><tr><td>
<a href="../docs.html">Back to the docs page</a><br>

<br>
<a href="generate.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="../tutorial/visualize.html">Next</a>



<h2>waffles_recommend</h2>
<p>
	A command-line tool for predicting missing values in incomplete data, or for
	testing collaborative filtering recommendation systems.
	Here's the usage information:</p>
<pre>
ENDTXT
waffles_recommend usage | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g' >> command/recommend.html
cat <<ENDTXT >> command/recommend.html
</pre>




<br>
<a href="generate.html">Previous</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="../tutorial/visualize.html">Next</a>

<br><br><a href="../docs.html">Back to the docs page</a>
</td></tr><tr><td background="../images/bar.png"><br>
</td></tr></table>
</body></html>
ENDTXT




