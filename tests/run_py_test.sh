#!/bin/bash

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_py_test.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../
GLIB_PATH="${CUR_PATH}"/../mx_rag/libs
export PYTHONPATH=$TOP_PATH:$PYTHONPATH:$GLIB_PATH

mkdir test_results

function run_test_cases() {
    echo "Get testcases final result."
    pytest --cov="${CUR_PATH}"/../mx_rag --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vs --cov-branch
    coverage xml -i --omit="build/*,cust_op/*,src/*,*/libs/*"
    cp coverage.xml final.xml final.html ./test_results
    cp -r htmlcov ./test_results
    rm -rf coverage.xml final.xml final.html htmlcov
}

pip3 install -r  ../requirements.txt
pip3 install langchain
echo "************************************* Start mxRAG LLT Test *************************************"
start=$(date +%s)
run_test_cases
ret=$?
end=$(date +%s)
echo "*************************************  End  mxRAG LLT Test *************************************"
echo "LLT running take: $(expr "${end}" - "${start}") seconds"


exit "${ret}"
