elename_lis = ['PLACE','PATH','SPATIAL_ENTITY','NONMOTION_EVENT','MOTION','SPATIAL_SIGNAL','MOTION_SIGNAL','MEASURE']
linkname_lis = ['QSLINK','OLINK','MOVELINK']

def remove_extra_spaces(text):

    new_text = ''
    oriidx2newidx=dict()
    remove_char=[' ','  ', '\n' ]
    
    idx=0
    while(idx<len(text) and text[idx] in remove_char):
        oriidx2newidx[idx]=-1
        idx+=1
    new_idx = 0
    while(idx<len(text)):
        while(idx<len(text) and text[idx]!=' ' and text[idx] not in remove_char):
            new_text += text[idx]
            oriidx2newidx[idx] = new_idx
            new_idx+=1
            idx+=1

        new_text += ' '
        oriidx2newidx[idx] = new_idx
        idx+=1
        new_idx+=1
        
        while(idx<len(text) and text[idx] in remove_char):
            oriidx2newidx[idx]=-1
            idx+=1
    
    return new_text, oriidx2newidx


if __name__ == '__main__':
    ori_text = '''Tokyo
        Originally known as Edo (meaning “estuary”), Tokyo was just
        a sleepy little village surrounded by marshland on the broad Kanto         plain until the end of the 16th century, when Tokugawa Ieyasu moved
        here and made it the center of his vast domains. When Ieyasu became
        shogun in 1603, Edo in turn became the seat of national
        government — and its castle the largest in the world. Edo expanded
        rapidly to accommodate Ieyasu’s 80,000 retainers and their families and
        the myriad common people who served their daily needs. By 1787 the
        population had grown to 1,368,000.
        The ruling elite lived on the high ground, the Yamanote
        (“bluffs”) west and south of the castle. The artisans, tradespeople,
        and providers of entertainment (reputable and not so reputable) lived
        “downtown” on the reclaimed marshlands north and east, in the area
        still known as Shitamachi. As these two populations interacted, a
        unique new culture was born. Edo became the center of power and also
        the center of all that was vibrant and compelling in the arts.
        After 1868 that center grew even stronger, when the movement
        known as the Meiji Restoration overthrew the Tokugawa shogunate and the
        imperial court moved to Edo. The city was renamed Tokyo (“Eastern         Capital”), and from that moment on all roads — political, cultural, and
        financial — led here.
        In the 20th century Tokyo has twice suffered almost total
        destruction. First, the earthquake of 1923 and subsequent fire razed
        nearly all vestiges of old Edo, killing some 140,000 people in the
        process. Rebuilt without any comprehensive urban plan, Tokyo remains a
        city of subcenters and neighborhoods, even villages, each with its own
        distinct personality.
        Unlike the great capital cities of Europe, there is no
        prevailing style of architecture here, no “monumental” core for a new
        building to harmonize or clash with. Even after the collapse of the
        economic bubble in 1992, construction projects are everywhere. Whole
        blocks of the city seem to disappear overnight, replaced in the blink
        of an eye by new office buildings, condominiums, cultural complexes,
        and shopping centers.
        Tokyo is a city of enormous creative and entrepreneurial
        energy, much of which goes into reinventing itself. If there’s a
        commodity in short supply here, it’s relaxation. Nobody “strolls” in
        Tokyo, and there are few places to sit down outdoors and watch the
        world go by. The idea of a long, leisurely lunch hour is utterly alien.
        People in Tokyo are in a hurry to get somewhere — even if they don’t
        always know precisely where they’re going. '''
    
    new_text, oriidx2newidx = remove_extra_spaces(ori_text)

    print(new_text)