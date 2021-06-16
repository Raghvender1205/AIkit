import random
import unittest
from unittest.main import main

import aikit.elastic

class Init(unittest.TestCase):
    def testStatics(self):
        global_batch_size = random.randint(10, 100)
        max_gpu_batch_size = random.randint(1, global_batch_size) # must be less than 'global_batch_size' because we don't support uneven spread
        gpus = 1 # currently multi-gpu is not supported in tests

        aikit.elastic._init(
            global_batch_size=global_batch_size,
            max_gpu_batch_size=max_gpu_batch_size,
            gpus=1
        )

        self.assertEqual(aikit.elastic.global_batch_size, global_batch_size)
        self.assertEqual(aikit.elastic.gpus, gpus)
        self.assertEqual(aikit.elastic.master, True)

        self.assertFalse(hasattr(aikit.elastic, 'hvd'))

if __name__ == '__main__':
    unittest.main()