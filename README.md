```bash
"""
# Exported for capture the meory

export PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT=1
export PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE=0
export PT_HPU_EAGER_PIPELINE_ENABLE=0
export PT_HPU_DISABLE_ASYNC_COLLECTIVE=1
export PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD=0
export PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD=0
export PT_HPU_ENABLE_COMPILE_THREAD=0
export PT_HPU_ENABLE_EXECUTION_THREAD=0
export PT_HPU_LAZY_ACC_PAR_MODE=0
export PT_HPU_SYNC_LAUNCH=1


export PT_TOWL_LOG_ENABLE=1
export HABANA_LOGS=.habana_logs-pp-mem-30

"""
"""
# Register the memory interceptor
import towl.instrument as ti
class HpuModelAdapter:

    def __init__(self, model, vllm_config, layer_names):
        # ti.MemoryInterceptor.install_wrappers_on(model,  recursive=True)
        
        ti.MemoryInterceptor.enable(0.5)
"""

"""
# Post process the log file
python -m towl.db create from-log-file /mnt/disk3/yiliu4/vllm-fork/scripts/.habana_logs-pp-mem-30/0/towl_log.txt -o ./pp-mem-db/test30/rank0 --overwrite
python pp_mem_tracker.py --rank 0 --test 30 --start 460
"""
```

Torch OWL (towl)
=================

Torch OWL is a tool for tracing memory allocations with PyTorch and Intel® Gaudi® AI accelerators.

The project contains the following Python subpackages:

* `towl-db` - Database interface that contains database definition and procedures to build one from collected traces.
* `towl-user` - End user interface that can be used with Jupyter notebook. Requires `towl-db`.
* `towl-instrument` - Instrumentation that contains various helpers to instrument topology and gather more data during runtime. It is a minimal dependency package that helps to prevent errors in your runtime environment.

We are currently in the early stages of developing the tool and expect to make significant changes to improve its usability in the near future.

## Instaling Packages from Sources

To install packages from a source, run `pip install .` in an suitable directory.

For example:

```
cd towl-user
pip install .
```

A helper for the `towl-db` and `towl-user` packages installation is located inside `Makefile`.

## Building Wheel Packages

You can build wheel packages using `poetry` Python build system. 
A helper for running build process on every Python subpackage is located inside `Makefile`.

For example:

```
make build
```

## Building API Documentation

You can build API documentation using pdoc3.
A helper for running build process on every Python subpackage is located inside `Makefile`.

For example:

```
make docs
```

## Usage and Examples

Refer to `examples/00_introduction` notebook.


## Comming Soon

Documentation updates and tool enhancements.
