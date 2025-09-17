#!/bin/bash

set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_py_test.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../
FAKE_PACKAGE_PATH="${CUR_PATH}"/fake_package
export PYTHONPATH=$TOP_PATH:$PYTHONPATH:$FAKE_PACKAGE_PATH

mkdir test_results

function run_test_cases() {
    echo "Get testcases final result."
    /opt/buildtools/python-3.11.4/bin/pytest --cov="${CUR_PATH}"/../mx_rag --cov-report=html --cov-report=xml --junit-xml=./final.xml --html=./final.html --self-contained-html --durations=5 -vs --cov-branch  --cov-config=.coveragerc
    coverage xml -i --omit="build/*,cust_op/*,src/*,*/libs/*,*/evaluate/*,*/train_data_generator.py,*/ops/*"
    cp coverage.xml final.xml final.html ./test_results
    cp -r htmlcov ./test_results
    rm -rf coverage.xml final.xml final.html htmlcov
}

pip3.11 install -r ../requirements.txt --exists-action i
echo "************************************* Start mxRAG LLT Test *************************************"
start=$(date +%s)
run_test_cases
ret=$?
end=$(date +%s)
echo "*************************************  End  mxRAG LLT Test *************************************"
echo "LLT running take: $(expr "${end}" - "${start}") seconds"


exit "${ret}"
