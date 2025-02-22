import os
import time


from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import NUM_REs


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    start_time = time.time()
    trainer = DeepSICTrainer(NUM_REs)
    print(trainer)
    trainer.evaluate()
    end_time = time.time()
    elapsed_time = end_time - start_time
    days = int(elapsed_time // (24 * 3600))
    elapsed_time %= 24 * 3600
    hours = int(elapsed_time // 3600)
    elapsed_time %= 3600
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

