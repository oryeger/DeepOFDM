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
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

