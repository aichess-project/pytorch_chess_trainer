from converter.convert_fen_halfkp import Convert_FEN_HalfKP2Simple
from libs.chess_lib import Chess_Lib

class Convert_FEN_Bitmap(Convert_FEN_HalfKP2Simple):
    
    def get_input_tensor(self, x, norm = True):
        return Chess_Lib.get_krk_bitmaps(x[0])