import chess
import torch
import logging

class Chess_Lib:
  piece_list = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
  color_list = [chess.WHITE, chess.BLACK]

  MAX_EVAL = (1000 * 2 + 100 * 1000)
  MAX_MATE = 100

  def get_eval_value(centi_pawn, mate_in):
    if mate_in == 0:
      return int(centi_pawn*100)
    if mate_in < 0:
      sign = -1
      mate_in = mate_in * sign
    else:
      sign = 1
    mate_in = min(mate_in, 99)
    mate_value = sign * (Chess_Lib.MAX_EVAL - mate_in * 1000)
    return mate_value
  
  def extract_turn_from_fen(fen):
    # Split the FEN string into its components
    parts = fen.split()

    # Extract the turn information (it's the first character of the second field)
    turn = parts[1][0]
    if turn == "w":
      return chess.WHITE
    else:
      return chess.BLACK
  
  def get_square_index(row, column):
     return row * 8 + column
  
  def normalize(pos):
        return pos/63.0
  
  def extract_krk_positions(fen, norm = False):
    # Split the FEN string to extract the piece placement part
    parts = fen.split(' ')
    piece_placement = parts[0]

    # Find the positions of white king, black king, and white queen
    white_king_position = 0
    black_king_position = 0
    white_rook_position = 0
    for row, row_str in enumerate(piece_placement.split('/')):
        col = 0
        for char in row_str:
            if char.isdigit() == False:
                if char == 'k':
                    black_king_position = Chess_Lib.get_square_index(7 - row, col)  # FEN uses rank 8 at the top
                elif char == 'K':
                    white_king_position = Chess_Lib.get_square_index(7 - row, col)
                elif char == 'R':
                    white_rook_position = Chess_Lib.get_square_index(7 - row, col)
                col += 1
            else:
                col += int(char)
    if norm:
       white_king_position = Chess_Lib.normalize(white_king_position)
       black_king_position = Chess_Lib.normalize(black_king_position)
       white_rook_position = Chess_Lib.normalize(white_rook_position)
    return white_king_position, black_king_position, white_rook_position
  
  def get_krk_bitmaps(fen):
    logging.debug(f"Get KRK Bitmap for: {fen}")
    white_king_position, black_king_position, white_rook_position = Chess_Lib.extract_krk_positions(fen, norm = False)
    turn = Chess_Lib.extract_turn_from_fen(fen)
    bitmap = torch.zeros(3*64+1, dtype = torch.bool)
    bitmap[white_king_position] = 1
    bitmap[white_rook_position + 64] = 1
    bitmap[black_king_position + 2*64] = 1
    if turn == chess.WHITE:
       bitmap[3*64] = 1
    logging.debug(bitmap)
    return bitmap.float()