#!/bin/bash
set -u -e # bail out on error
if [ -e apidoc/html ]; then
    echo "Removing old documentation files."
    rm -f apidoc/html/*
fi
doxygen ./doxygen_config.txt
#echo '<br><br><br><br><br><br><br><br><br>Footer' >> apidoc/html/tree.html
