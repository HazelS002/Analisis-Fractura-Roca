import numpy as np


def get_stage_images(results:list[dict], stage:str) -> list[np.ndarray]:
    return [result[stage] for result in results]