#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Description: SDK uninstallation tool.
# Author: Mind SDK
# Create: 2024
# History: NA
set -e

# Simple log helper functions
CUR_PATH=$(cd "$(dirname "$0")" || { echo "Failed to enter current path" ; exit ; } ; pwd)

PACKAGE_LOG_NAME=mxRag
info_record_path="${HOME}/log/mxRag"
info_record_file="deployment.log"
info_record_file_back="deployment.log.bak"
log_file=$info_record_path/$info_record_file
LOG_SIZE_THRESHOLD=1024000

readonly USER_N="$(whoami)"
readonly WHO_PATH="$(which who)"
readonly CUT_PATH="$(which cut)"
IP_N="$("${WHO_PATH}" -m | "${CUT_PATH}" -d '(' -f 2 | "${CUT_PATH}" -d ')' -f 1)"
if [ "${IP_N}" = "" ]; then
   IP_N="localhost"
fi

function rotate_log() {
    check_path "$log_file"
    mv -f "$log_file" "$info_record_path/$info_record_file_back"
    touch "$log_file" 2>/dev/null
    check_path "$info_record_path/$info_record_file_back"
    chmod 440 "$info_record_path/$info_record_file_back"
    check_path "$log_file"
    chmod 640 "$log_file"
}

function check_path() {
    if [ "$1" != "$(realpath "$1")" ]; then
      echo
      echo "Log file is not support symlink, exiting!"
      exit 1
    fi
}

function log_check() {
    local log_size
    log_size="$(find $log_file -exec ls -l {} \; | awk '{ print $5 }')"
    if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
        rotate_log
    fi
}
# usage log "INFO" "this is message"
function log() {
    # print log to log file
    if [ "$log_file" = "" ]; then
        echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$USER_N] [$IP_N]:\n$1" >&2
    fi
    if [ -f "$log_file" ]; then
        log_check "$log_file"
        if ! echo -e "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [$USER_N] [$IP_N]:\n$1" >>"$log_file"
        then
          echo "Can not write log, exiting!"
          exit 1
        fi
    else
        echo "Log file does not exist, exiting!"
        exit 1
    fi
}

get_run_path() {
  run_path=$(pwd)
  cd ..
  if [[ "$run_path" =~ '/mxRag' ]];then
    suffix='mxRag'
  else
    echo "Directory mxRag does not exist in path[$(pwd)], exiting!"
    exit 1
  fi
  del_path=$(pwd)/"$suffix"
}
real_delete() {
  cd "${CUR_PATH}/.." || {
    echo "Where is the RAG SDK?"
    exit 255
  }
  get_run_path
  if [[ -f "$del_path"/version.info ]];then
    version_info="$del_path"/version.info
    if [ ! -d "$info_record_path" ];then
        mkdir -p "$info_record_path"
        chmod 750 "$info_record_path"
    fi

    if [[ ! -f "$log_file" ]];then
        touch "$log_file"
    fi
    find "$log_file" -type f -exec chmod 640 {} +
    log "$(cat "${version_info}")"

    python3 -m pip uninstall mx-rag -y

    chmod u+w -R "$del_path"
    del_real_path=$(realpath "$del_path")
    # remove soft link first
    if [ -d "$del_path" ];then
      rm -rf "$del_path"
    fi
    # remove real files
    if [ -d "$del_real_path" ];then
      rm -rf "$del_real_path"
    fi

    log "Uninstall RAG SDK package successfully."
    echo 'Uninstall RAG SDK package successfully.'
  fi
}

real_delete
