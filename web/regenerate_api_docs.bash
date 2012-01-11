#!/bin/bash
set -u -e # bail out on error
if [ -e apidoc/html ]; then
    echo "Removing old documentation files."
    rm -f apidoc/html/*
fi
doxygen ./doxygen_config.txt
echo '<br><br><br><br><br><br><br><br><br><br><br><a href="http://sourceforge.net"><img src="http://sourceforge.net/sflogo.php?group_id=153538&amp;type=3" width="125" height="37" border="0" alt="SourceForge.net Logo" /></a>' >> apidoc/html/tree.html
