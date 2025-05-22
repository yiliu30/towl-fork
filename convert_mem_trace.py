# # 

# python -m towl.db create from-log-file /mnt/disk3/yiliu4/vllm-fork/scripts/.habana_logs-pp-mem-22-5/7/towl_log.txt -o ./a-pp-mem-db/test5/rank7 --overwrite

import subprocess
import os
import argparse

arg = argparse.ArgumentParser()
arg.add_argument('--test', type=int, default=0)

args = arg.parse_args()
test = args.test


log_base_dir = f"/mnt/disk3/yiliu4/vllm-fork/scripts/.habana_logs-pp-mem-22-{test}"
output_base_dir = f"./a-pp-mem-db/test{test}"
num_ranks = 8  # adjust this if you have a different number of ranks

for rank in [0, 7]:
    log_path = os.path.join(log_base_dir, str(rank), "towl_log.txt")
    output_path = os.path.join(output_base_dir, f"rank{rank}")

    cmd = [
        "python", "-m", "towl.db",
        "create", "from-log-file", log_path,
        "-o", output_path,
        "--overwrite"
    ]

    print(f"Running for rank {rank}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
