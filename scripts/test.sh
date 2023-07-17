#!/bin/bash

# clear
# set -x

# log dir
ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# set HIP_VISIBLE_DEVICES=0

# export MLIR_ENABLE_DUMP=1
# export LLVM_IR_ENABLE_DUMP=1
# export AMDGCN_ENABLE_DUMP=1

# UNIT_TEST="test_amd.py"
# UNIT_TEST="test_jax.py"
# UNIT_TEST="tests"
# UNIT_TEST="tests/pallas_test.py"
# UNIT_TEST="tests/pallas_test_med.py"
# UNIT_TEST="tests/pallas_test.py::PallasCallTest::test_add_matrix_block_spec"
UNIT_TEST="tests/triton_call_test.py"
# UNIT_TEST="tests/triton_call_test.py::TritonKernelCallTest::test_add0"
# UNIT_TEST="tests/triton_test.py"
# UNIT_TEST="tests/triton_test.py::TritonTest::test_add_kernel"
# UNIT_TEST="tests/triton_test.py::TritonTest::test_tanh_kernel"

LOG_FILE_NAME=$(basename $UNIT_TEST)

OUTPUT_MODE="--capture=tee-sys -v"
# OUTPUT_MODE="--capture=tee-sys -vv"

# check for backtrace
if [ "$1" == "backtrace" ]; then
    sudo apt install gdb -y

    COMMAND="-m pytest $OUTPUT_MODE $UNIT_TEST"
    gdb python \
        -ex "set pagination off" \
        -ex "run $COMMAND" \
        -ex "backtrace" \
        -ex "set confirm off" \
        -ex "q" \
        2>&1 | tee $LOG_DIR/${LOG_FILE_NAME}_backtrace.log

else
    pytest $OUTPUT_MODE -rfs --verbose "$UNIT_TEST" 2>&1 | tee $LOG_DIR/${LOG_FILE_NAME}.log
fi
