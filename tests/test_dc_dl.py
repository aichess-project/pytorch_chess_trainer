from data_classes.dc_dataloader import DC_DataLoad
import unittest, sys
from converter.convert_fen_halfkp import Convert_FEN_HalfKP2Simple

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

    def test_testloader(self):
        data_path = r"C:\Users\littl\Documents\GIT\AI\krk_data\data\test\KRk_new.csv"
        print(data_path)
        delimiter = ","
        converter = Convert_FEN_HalfKP2Simple()
        xl = []
        yl = []
        try:
            with open(data_path, 'r') as file:
                lines = file.readlines()
                print (lines)
                print(f"Length Lines: {len(lines)}")
                for line in lines:
                    # Split the line using the specified delimiter
                    print (line)
                    items = line.strip().split(delimiter)
                    x = []
                    for index, item in enumerate(items):
                        if index == len(items)-1:
                            y = float(item)
                        else:
                            x.append(item)
                    y = converter.get_output_tensor(y)
                    x = converter.get_input_tensor(x, norm = False)
                    print(f"X: {x}")
                    print(f"Y: {y}")
                    yl.append(y)
                    xl.append(x)
                #print(yl)
                #print(xl)
        except Exception as e:
            print.info(f"Exception: {e}")
            sys.exit()

def main():
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestDC_DL))

if __name__ == "__main__":
    main()
