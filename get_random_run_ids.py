import wandb

# parser=argparse.ArgumentParser(description="Process some input files")
# parser.add_argument('--run_id', help='which run should be edited')
# parser.add_argument('--param', help='parameter to change')
# parser.add_argument('--param_type', help='typecast of param, such as int or str', type=str)
# parser.add_argument('--value', help='value to change param to')
# args = parser.parse_args()

# run_id = args.run_id
api = wandb.Api()
runs = api.runs(path="jdonovan/novel-feature-detectors", filters={"config.experiment_name": "random normal", "config.fixed_conv": False, "config.dataset": "cifar-100"})#, "config.evo": 0, "config.fixed_conv": 0})

def confirm(prompt=None, resp=False):
    """prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.

    >>> confirm(prompt='Create Directory?', resp=True)
    Create Directory? [y]|n: 
    True
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: 
    False
    >>> confirm(prompt='Create Directory?', resp=False)
    Create Directory? [n]|y: y
    True

    """
    
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
        
    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

# data = []
# with open('artifacts/run-um0c4jr0-history-v0/0000.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
#     for row in spamreader:
#         data.append(row)
# print(data)
# table = wandb.Table(columns = data[0], data=data[1:])

# run = api.run(path='jdonovan/novel-feature-detectors/3doqm1v1')
# for artifact in run.logged_artifacts():
#     if artifact.type == "model":
#         artifact.delete(delete_aliases=True)
#     if artifact.type == "wandb-history":
#         print(dir(artifact))
#         print(dir(artifact._files))
#         print((artifact.manifest.to_manifest_json()))
#         artifact.add(table, "testing_update")
# artifact = api.artifact('jdonovan/novel-feature-detectors/run-3doqm1v1-history:v0', type='wandb-history')
# artifact.delete()


for run in runs:
    print(run.id)
    # print([type(row['val_novelty']) for row in history])
    # values = [row['val_novelty'] for row in history]
    # if np.mean(values) < .2:
    #     values = np.array(values)*6
    # values2 = [row['val_novelty_epoch'] for row in history]
    # if np.mean(values2) < .2:
    #     values2 = np.array(values)*6
    # run.summary["val_novelty_corrected"] = values
    # run.summary["val_novelty_epoch_corrected"] = values2
# if not args.value:
#     if confirm("Replace the value for this parameter with null / None?", True):
#         value = None
# elif args.param_type == "int":
#     value = int(args.value)
# elif args.param_type == "float":
#     value = float(args.value)
# elif args.param_type == "bool":
#     value = bool(int(args.value))
# else:
#     value = str(args.value)
    # run.summary.update()