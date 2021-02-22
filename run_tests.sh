#!/bin/bash
set -e
set -o pipefail

make setup_tests -j8

while read line
do
  ./$line
done < tmpfiles.txt
rm tmpfiles.txt
