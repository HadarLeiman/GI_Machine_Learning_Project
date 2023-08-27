# Import the W&B Python Library and log into W&B
import wandb
from main_for_wrist import main_wrist

wandb.login()


def main():
    wandb.init(project='first-sweep-try')
    test_acc = main_wrist(wandb.config)
    wandb.log({"Test accuracy": test_acc})


# 2: Define the search space
sweep_configuration = {
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'Test accuracy'},
    'parameters':
        {
            'lr': {'values': [0.00001, 0.00002, 0.00003]},
            'epoch': {'values': [10, 20, 50, 100]},
            'shape': {'values': [(32, 64), (64, 128), (128, 256)]},
            'num_of_measurements': {'values': [512, 1024, 2048, 4096, 8192]},
            'batch_size': {'values': [16, 32, 64, 128]},
        }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='first-sweep-try')
wandb.agent(sweep_id, function=main)
