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
import argparse

import os
arg = argparse.ArgumentParser()
arg.add_argument('--rank', type=int, default=0)
arg.add_argument('--test', type=int, default=0)
arg.add_argument('--start', type=int, default=0)
arg.add_argument('--end', type=int, default=None)
args = arg.parse_args()
rank = args.rank
test = args.test
end = args.end
start = args.start

# %matplotlib inline
import matplotlib.pyplot as plt



# Good package for browsing DataFrames
import itables

# Configuring matplotlib output size
plt.rcParams['figure.figsize'] = (12, 5)


import towl.user as tu

file_name = f"./pp-mem-db/test{test}/rank{rank}"

scenario = tu.Scenario(file_name)

print('Scenario global time range:', scenario.global_event_timerange)
# breakpoint()
# event_filter = scenario.global_event_timerange.make(279, scenario.global_event_timerange.end)
# event_filter = scenario.global_event_timerange.make(370, scenario.global_event_timerange.end)
# start, end  = 0, 255
# start, end  = 450, scenario.global_event_timerange.end
if end is None:
    end = scenario.global_event_timerange.end
result_file = f"mem_test{test}_rank{rank}_{start}_{end}.pickle"
html_file = f"mem_test{test}_rank{rank}_{start}_{end}.html"
event_filter = scenario.global_event_timerange.make(start, end)
# event_filter = scenario.global_event_timerange
global_view = scenario.make_view(event_filter)


# # shortcut:
# global_view = scenario.make_global_view()
# tu.plots.plot_memory_usage(global_view)
global_memory_usage_df = global_view.query_memory_usage()
print(type(global_memory_usage_df))

# print global_memory_usage_df col name
print(global_memory_usage_df.columns)
print(global_memory_usage_df.head())

# df to csv
# global_memory_usage_df.to_csv('global_memory_usage_df.csv')
global_hills = tu.lib.find_hills(global_memory_usage_df['used'], 1000)
print('Found', len(global_hills), 'hills')
print('Hill 0 =', global_hills[0])
print('Hill 1 =', global_hills[1])
print('Hill -1 =', global_hills[-1])
# for i in range(100):
#     print('Hill', i, '=', global_hills[i])

hill_view = scenario.make_view(global_hills[0])
tu.lib.dump_cudamemviz(global_view, result_file, html_file)

"""
Building snapshot: 100%|█████████████████| 2099/2099 [00:00<00:00, 19109.84entry/s]
> /usr/bin/python -m torch.cuda._memory_viz trace_plot -o 'global_decode_test_30_rank0_460_2959.html' 'global_decode_test_30_rank0_460_2959.pickle'
Calling add_step_closure function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
Calling iter_mark_step function does not have any effect. It's lazy mode only functionality. (warning logged once)
/usr/lib/python3.10/runpy.py:126: RuntimeWarning: 'torch.cuda._memory_viz' found in sys.modules after import of package 'torch.cuda', but prior to execution of 'torch.cuda._memory_viz'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
"""
