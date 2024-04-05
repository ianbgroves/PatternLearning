import psutil
import os
import wandb

def check_memory_usage(prepend=None):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    if prepend is not None:
        return print(f"{prepend} Memory usage: {memory_info.rss / 1024 ** 2} MB")
    else:
        return print(f"Memory usage: {memory_info.rss / 1024 ** 2} MB")

def wandb_setup(project, entity, name, config=wandb.config):
    wandb.init(project=project, entity=entity, name=name, config=config)
    return wandb.config