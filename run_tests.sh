#!/bin/bash
set -e
set -o pipefail

make setup_tests -j2

while read line
do
  echo ./$line
  ./$line
done < tmpfiles.txt
rm tmpfiles.txt
