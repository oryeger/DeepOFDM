import os

from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import NUM_REs


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    trainer = DeepSICTrainer(NUM_REs)
    print(trainer)
    trainer.evaluate()
