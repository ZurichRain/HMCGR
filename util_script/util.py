def get_char2tok_spanlis_one_seq(seq,tokenizer):
    '''
        used for get char-index to token-index
        charidx = [tokst,toked) 
    '''
    token_span = tokenizer.encode_plus(seq, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    char_num = None
    for tok_ind in range(len(token_span) - 1, -1, -1):
        if token_span[tok_ind][1] != 0:
            char_num = token_span[tok_ind][1]
            break
    char2tok_span = [[-1, -1] for _ in range(char_num)]
    for tok_ind, char_sp in enumerate(token_span):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            if tok_sp[0] == -1:
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1
    return char2tok_span

if __name__ == '__main__':
    pass