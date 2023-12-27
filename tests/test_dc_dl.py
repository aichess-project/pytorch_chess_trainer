from data_classes.dc_dataloader import DC_DataLoad
import unittest

class TestDC_DL(unittest.TestCase):

    def setUp(self):
        # Code to run before each test
        pass

    def tearDown(self):
        # Code to run after each test
        pass

    def test_dataloaders(self):

        types = DC_DataLoad.get_all_types()
        self.assertEqual(len(types), 3)
        subdir = DC_DataLoad.get_sub_dir("training")
        self.assertEqual(subdir, "train")
        is_training = DC_DataLoad.training()
        self.assertEqual(is_training, "training")

def main():
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestDC_DL))

if __name__ == "__main__":
    main()
