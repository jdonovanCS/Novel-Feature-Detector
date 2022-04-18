import wandb
import argparse

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_id', help='which run should be edited')
parser.add_argument('--param', help='parameter to change')
parser.add_argument('--param_type', help='typecast of param, such as int or str', type=str)
parser.add_argument('--value', help='value to change param to')
args = parser.parse_args()

run_id = args.run_id
api = wandb.Api()
run = api.run("jdonovan/novel-feature-detectors/" + run_id)
if args.param_type == "int":
    value = int(args.value)
if args.param_type == "float":
    value = float(args.value)
else:
    value = str(args.value)
run.config[args.param] = value
run.update()