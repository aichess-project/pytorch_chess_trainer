from converter.converter_base import Converter_Base
from libs.chess_lib import Chess_Lib
import torch

class Convert_FEN_HalfKP2Simple(Converter_Base):
    
    def get_input_tensor(self, x, norm = True):
        turn = Chess_Lib.extract_turn_from_fen(x[0])
        white_king_position, black_king_position, white_rook_position = Chess_Lib.extract_krk_positions(x[0], norm = norm)
        return torch.tensor([white_king_position, black_king_position, white_rook_position, turn], dtype=torch.float)

    def get_output_tensor(self, y):
        return torch.tensor([y], dtype=torch.float)